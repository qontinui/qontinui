"""Multi-element proposal strategy.

v5 is a single-point grounder. This module wraps it with the milestone-a
strategy: iterate over a fixed list of UI categories (Button, Input, Tab,
Link, Icon, Label) and ask v5 "Locate the most prominent {category}".
Results whose centers are within 20 px of each other are deduplicated.

A future experiment is to replace the iteration with a single "enumerate
interactive elements" prompt; the plan explicitly defers that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from .client import VgaClient, VgaClientError
from .prompts import DEFAULT_PROPOSAL_CATEGORIES, PROPOSE_CATEGORY_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

_DEDUP_PX_THRESHOLD = 20
"""Centers within this many pixels are treated as the same proposal."""


@dataclass(frozen=True)
class ElementProposal:
    """A single proposed UI element.

    Attributes:
        label: Human-readable label, defaults to the category name.
        prompt: The exact prompt phrase that would re-ground this element
            at runtime (e.g. ``"Button"``). The builder UI lets the user
            refine this before saving.
        x: Pixel x of the element's predicted center.
        y: Pixel y.
        confidence: Heuristic confidence from the client (0-1).
        category: Which category query produced this proposal.
    """

    label: str
    prompt: str
    x: int
    y: int
    confidence: float
    category: str


class ElementProposer:
    """Produces a deduplicated list of :class:`ElementProposal` objects.

    Args:
        client: A :class:`~qontinui.vga.client.VgaClient` to query. The
            proposer owns no network config of its own.
    """

    def __init__(self, client: VgaClient) -> None:
        self._client = client

    def propose(
        self,
        image: Any,
        categories: list[str] | None = None,
    ) -> list[ElementProposal]:
        """Return proposals for each category that the model can locate.

        Args:
            image: PIL Image / numpy array / raw bytes.
            categories: Override the default category list. ``None`` uses
                :data:`qontinui.vga.prompts.DEFAULT_PROPOSAL_CATEGORIES`.

        Returns:
            A list of proposals, deduplicated by pixel proximity.
        """
        cats = list(categories) if categories is not None else list(DEFAULT_PROPOSAL_CATEGORIES)
        proposals: list[ElementProposal] = []

        for category in cats:
            try:
                result = self._client.ground(
                    image,
                    category,
                    prompt_template=PROPOSE_CATEGORY_PROMPT_TEMPLATE,
                )
            except VgaClientError:
                logger.debug("proposer: client error for category=%s", category, exc_info=True)
                continue

            if result.confidence <= 0.0:
                # Model returned <none/> or an unparseable response.
                continue

            proposals.append(
                ElementProposal(
                    label=category,
                    prompt=category,
                    x=result.x,
                    y=result.y,
                    confidence=result.confidence,
                    category=category,
                )
            )

        return self._dedupe(proposals)

    @staticmethod
    def _dedupe(proposals: list[ElementProposal]) -> list[ElementProposal]:
        """Drop proposals whose centers are within 20 px of a prior one.

        Keeps the higher-confidence proposal on collision. Stable order
        otherwise.
        """
        kept: list[ElementProposal] = []
        for candidate in proposals:
            dup_idx: int | None = None
            for idx, existing in enumerate(kept):
                dx = candidate.x - existing.x
                dy = candidate.y - existing.y
                if math.hypot(dx, dy) <= _DEDUP_PX_THRESHOLD:
                    dup_idx = idx
                    break

            if dup_idx is None:
                kept.append(candidate)
            elif candidate.confidence > kept[dup_idx].confidence:
                kept[dup_idx] = candidate

        return kept
