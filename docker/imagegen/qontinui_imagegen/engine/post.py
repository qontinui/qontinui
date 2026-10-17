"""The named runners: Real-ESRGAN upscaling, BiRefNet / BEN2 matting, vtracer tracing.

Each runs the pinned code the manifest lists for it (``runners[].code``): the vendored model
code under ``docker/imagegen/vendor`` and the pinned libraries. Weights load from local
files only, and torch pickles only with ``weights_only=True`` (manifest ``runtime``).
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from typing import Any

from qontinui_imagegen.config import FacadeConfig
from qontinui_imagegen.jobs import Catalogue, PostStep
from qontinui_imagegen.weights import component_dir

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MATTE_SIZE = 1024


class PostRunners:
    def __init__(self, catalogue: Catalogue, config: FacadeConfig) -> None:
        self.catalogue = catalogue
        self.config = config
        self._models: dict[str, Any] = {}

    def release(self) -> None:
        self._models.clear()

    def _folder(self, model_id: str) -> Any:
        return component_dir(self.config.weights_root, self.catalogue.components[model_id])

    def apply(self, step: PostStep, image: Any) -> Any:
        if step.op == "upscale":
            return self.upscale(step.models[0], image, step.parameters)
        if step.op == "matte":
            return self.matte(step.runner, step.models[0], image)
        if step.op == "vectorize":
            return vectorize(image, step.parameters)
        raise ValueError(f"no runner for post op {step.op!r}")

    # --- Real-ESRGAN --------------------------------------------------------------------------

    def _esrgan(self, model_id: str) -> tuple[Any, int]:
        import importlib

        import torch

        if model_id not in self._models:
            component = self.catalogue.components[model_id]
            architecture = component["architecture"]
            module_name = (
                architecture["module"].removeprefix("vendor/").removesuffix(".py").replace("/", ".")
            )
            network_cls = getattr(importlib.import_module(module_name), architecture["class"])
            network = network_cls(**architecture["kwargs"])
            path = self._folder(model_id) / component["files"][0]["path"]
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            state = next(
                (checkpoint[key] for key in architecture["state_dict_keys"] if key in checkpoint),
                checkpoint,
            )
            network.load_state_dict(state, strict=True)
            network.eval().to(self.config.device)
            self._models[model_id] = (network, int(architecture["netscale"]))
        network, scale = self._models[model_id]
        return network, scale

    def upscale(self, model_id: str, image: Any, parameters: Mapping[str, Any]) -> Any:
        """Real-ESRGAN tiled inference, following realesrgan/utils.py's RealESRGANer.

        An alpha channel is upscaled separately with bicubic resampling.
        """
        import numpy as np
        import torch
        from PIL import Image

        if float(parameters.get("denoise_strength", 1.0)) != 1.0:
            # Below 1.0 upstream interpolates with realesr-general-wdn-x4v3, which the
            # manifest does not pin; recipe upscale v1 pins 1.0 for that reason.
            raise ValueError("denoise_strength other than 1.0 needs an unpinned model")
        network, scale = self._esrgan(model_id)
        half = bool(parameters.get("half", True))
        network = network.half() if half else network.float()
        alpha = image.getchannel("A") if image.mode == "RGBA" else None
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.config.device)
        tensor = tensor.half() if half else tensor
        pre_pad = int(parameters.get("pre_pad", 0))
        if pre_pad:
            tensor = torch.nn.functional.pad(tensor, (0, pre_pad, 0, pre_pad), "reflect")
        mod_scale = 2 if scale == 2 else (4 if scale == 1 else None)
        mod_h = mod_w = 0
        if mod_scale is not None:
            _, _, height, width = tensor.shape
            mod_h = (mod_scale - height % mod_scale) % mod_scale
            mod_w = (mod_scale - width % mod_scale) % mod_scale
            tensor = torch.nn.functional.pad(tensor, (0, mod_w, 0, mod_h), "reflect")
        tile = int(parameters.get("tile", 0))
        with torch.no_grad():
            if tile > 0:
                output = _tile_process(
                    network, tensor, scale, tile, int(parameters.get("tile_pad", 10))
                )
            else:
                output = network(tensor)
        _, _, out_h, out_w = output.shape
        output = output[:, :, : out_h - mod_h * scale, : out_w - mod_w * scale]
        if pre_pad:
            _, _, out_h, out_w = output.shape
            output = output[:, :, : out_h - pre_pad * scale, : out_w - pre_pad * scale]
        array = output.squeeze(0).float().clamp_(0, 1).cpu().numpy().transpose(1, 2, 0)
        result = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
        if alpha is not None:
            result.putalpha(alpha.resize(result.size, Image.BICUBIC))
        return result

    # --- matting ---------------------------------------------------------------------------

    def matte(self, runner: str, model_id: str, image: Any) -> Any:
        import torch

        if runner == "birefnet":
            model = self._models.get(model_id)
            if model is None:
                from birefnet.birefnet import BiRefNet

                model = BiRefNet.from_pretrained(
                    str(self._folder(model_id)), local_files_only=True, use_safetensors=True
                )
                model = model.to(self.config.device).half().eval()
                self._models[model_id] = model
            return _birefnet_matte(model, image, self.config.device)
        if runner == "ben2":
            model = self._models.get(model_id)
            if model is None:
                from ben2.BEN2 import BEN_Base
                from safetensors.torch import load_file

                model = BEN_Base()
                model.load_state_dict(load_file(str(self._folder(model_id) / "model.safetensors")))
                model = model.to(self.config.device).eval()
                if torch.cuda.is_available():
                    model = model.half()  # BEN2's img_transform feeds fp16 on CUDA
                self._models[model_id] = model
            return model.inference(image.convert("RGB"))
        raise ValueError(f"no matte runner {runner!r}")


def _birefnet_matte(model: Any, image: Any, device: str) -> Any:
    """The card's inference: 1024x1024, ImageNet normalisation, sigmoid of the last map."""
    import numpy as np
    import torch
    from PIL import Image

    rgb = image.convert("RGB")
    resized = rgb.resize((MATTE_SIZE, MATTE_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32
    )
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device).half()
    with torch.no_grad():
        prediction = model(tensor)[-1].sigmoid().float().cpu()[0, 0].numpy()
    mask = Image.fromarray((prediction * 255.0).round().astype(np.uint8), mode="L").resize(
        rgb.size, Image.BILINEAR
    )
    result = rgb.copy()
    result.putalpha(mask)
    return result


def _tile_process(network: Any, tensor: Any, scale: int, tile: int, tile_pad: int) -> Any:
    """RealESRGANer.tile_process: overlapping tiles, each cropped back to its own area."""
    import torch

    batch, channels, height, width = tensor.shape
    output = tensor.new_zeros((batch, channels, height * scale, width * scale))
    tiles_x = math.ceil(width / tile)
    tiles_y = math.ceil(height / tile)
    for y in range(tiles_y):
        for x in range(tiles_x):
            start_x, start_y = x * tile, y * tile
            end_x, end_y = min(start_x + tile, width), min(start_y + tile, height)
            pad_start_x, pad_end_x = max(start_x - tile_pad, 0), min(end_x + tile_pad, width)
            pad_start_y, pad_end_y = max(start_y - tile_pad, 0), min(end_y + tile_pad, height)
            with torch.no_grad():
                piece = network(tensor[:, :, pad_start_y:pad_end_y, pad_start_x:pad_end_x])
            out_x0 = (start_x - pad_start_x) * scale
            out_y0 = (start_y - pad_start_y) * scale
            out_x1 = out_x0 + (end_x - start_x) * scale
            out_y1 = out_y0 + (end_y - start_y) * scale
            output[:, :, start_y * scale : end_y * scale, start_x * scale : end_x * scale] = piece[
                :, :, out_y0:out_y1, out_x0:out_x1
            ]
    return output


def vectorize(image: Any, parameters: Mapping[str, Any]) -> str:
    """Trace a raster (transparency kept) to SVG with vtracer's recipe-pinned parameters."""
    import vtracer

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return str(
        vtracer.convert_raw_image_to_svg(buffer.getvalue(), img_format="png", **dict(parameters))
    )
