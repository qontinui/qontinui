"""Prompt templates for the VGA grounding model.

Two templates:

- :data:`GROUND_PROMPT_TEMPLATE` — single-element grounding. This is the
  exact format that ``grounding_vlm_backend.py`` has been using against
  ``qontinui-grounding-v5``; keep the wording stable so v5 predictions
  stay consistent across call sites.
- :data:`PROPOSE_CATEGORY_PROMPT_TEMPLATE` — asks for the most prominent
  element matching a category (Button, Input, Tab, ...). Category-by-
  category iteration is the milestone (a) proposal strategy; a single
  "list all elements" prompt is a deferred experiment (see plan §4).
"""

from __future__ import annotations

GROUND_PROMPT_TEMPLATE = (
    "Locate the following element in the screenshot and output its "
    "position as <point>x y</point> where x and y are integers "
    "between 0 and 1000 (normalized coordinates).\n\n"
    "Element: {description}"
)
"""Single-element grounding prompt. ``{description}`` is a natural-language
element description, e.g. ``"Save button in the toolbar"``.

Kept intentionally identical to the prompt hard-coded in
``grounding_vlm_backend.py`` prior to the VgaClient extraction. v5 was
trained against this exact wording; changing it risks measurable
regressions until the next retrain.
"""


PROPOSE_CATEGORY_PROMPT_TEMPLATE = (
    "Locate the most prominent {category} in the screenshot and output "
    "its position as <point>x y</point> where x and y are integers "
    "between 0 and 1000 (normalized coordinates). If no {category} is "
    "clearly visible, respond with <none/>."
)
"""Per-category proposal prompt. ``{category}`` is one of the default
categories (``Button``, ``Input``, ``Tab``, ``Link``, ``Icon``,
``Label``) or any user-supplied category string."""


DEFAULT_PROPOSAL_CATEGORIES: tuple[str, ...] = (
    "Button",
    "Input",
    "Tab",
    "Link",
    "Icon",
    "Label",
)
"""Default categories iterated by :class:`qontinui.vga.proposer.ElementProposer`."""
