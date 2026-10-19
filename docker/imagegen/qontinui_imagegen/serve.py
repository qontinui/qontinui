"""Run the facade: ``python -m qontinui_imagegen.serve --host 127.0.0.1 --port <PORT>``.

llama-swap starts this as the ``qontinui-imagegen`` model's ``cmd`` (docker/llama-swap/
config.yaml) with ``${PORT}`` and a loopback host, and stops it after its ``ttl``. The host
must be a loopback address: the facade has no authentication of its own, and the only
authorization for image work on the box is that loopback bind (D1.6) -- a non-loopback host
is refused rather than served.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from qontinui_imagegen.admission import AdmissionController
from qontinui_imagegen.app import Facade, create_app
from qontinui_imagegen.config import load_config
from qontinui_imagegen.jobs import Catalogue
from qontinui_imagegen.manifest import MANIFEST_PATH, RECIPES_DIR, check, load_recipes, load_yaml
from qontinui_imagegen.weights import WeightVerifier


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qontinui_imagegen.serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    if not _loopback(args.host):
        print(
            f"refusing to listen on {args.host!r}: the facade binds loopback only (D1.6)",
            file=sys.stderr,
        )
        return 2
    # The runtime contract every load follows (manifest.yaml `runtime`): no network fetch.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    violations = check(MANIFEST_PATH, RECIPES_DIR)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        print("refusing to serve: the manifest or a recipe is not admitted", file=sys.stderr)
        return 2
    manifest = load_yaml(MANIFEST_PATH)
    recipes = load_recipes(RECIPES_DIR)
    models = {str(c["id"]) for c in manifest["components"] if c.get("kind") == "model"}
    config = load_config(args.config, known_models=models, known_recipes=recipes)
    if config.admission.bulk_window_timezone is None:
        # Not a startup error (admission.py says why), but never a silent one either.
        print(
            "notice: no bulk-window timezone is set (facade.yaml admission.bulk_window."
            "timezone or QONTINUI_IMAGEGEN_TIMEZONE), so every bulk request is answered "
            "`deferred` (bulk_window_timezone_unset); interactive work is served",
            file=sys.stderr,
        )
    catalogue = Catalogue(config, manifest, recipes)
    verifier = WeightVerifier(
        config.weights_root,
        catalogue.components,
        config.weights_root / config.stamps_file,
        config.stamp_fields,
    )
    from qontinui_imagegen.engine.diffusers_engine import DiffusersEngine

    engine = DiffusersEngine(catalogue, config)
    facade = Facade(catalogue, engine, verifier, AdmissionController(config.admission))
    import uvicorn

    uvicorn.run(
        create_app(facade), host=args.host, port=args.port, log_level="info", access_log=False
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
