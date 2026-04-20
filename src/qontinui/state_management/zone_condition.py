"""Zone-based condition evaluator for state machine transitions.

Evaluates whether detection counts within a ``PolygonZone`` meet a
threshold, enabling zone-based state transition triggers.  For example,
"transition to INVENTORY_FULL when >= 28 items detected in the
inventory grid zone".

Example::

    from qontinui.find.zones import PolygonZone, Position
    from qontinui.state_management.zone_condition import (
        ZoneCondition,
        ZoneConditionEvaluator,
    )

    zone = PolygonZone(
        polygon=np.array([[100, 100], [500, 100], [500, 400], [100, 400]]),
        triggering_position=Position.CENTER,
    )
    condition = ZoneCondition(
        zone=zone,
        operator=">=",
        count_threshold=28,
    )
    evaluator = ZoneConditionEvaluator()
    evaluator.register("inventory_zone", condition)

    # Each frame, feed detections and check if transitions should fire
    triggered = evaluator.evaluate_all(detections)
    # triggered == {"inventory_zone": True/False}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..find.detections import Detections
    from ..find.zones import PolygonZone

logger = logging.getLogger(__name__)


class ComparisonOperator(Enum):
    """Comparison operators for zone count thresholds."""

    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="


_OP_FNS = {
    ComparisonOperator.EQ: lambda a, b: a == b,
    ComparisonOperator.NE: lambda a, b: a != b,
    ComparisonOperator.GT: lambda a, b: a > b,
    ComparisonOperator.GE: lambda a, b: a >= b,
    ComparisonOperator.LT: lambda a, b: a < b,
    ComparisonOperator.LE: lambda a, b: a <= b,
}

# Convenience map from string operator to enum
_STR_TO_OP = {op.value: op for op in ComparisonOperator}


@dataclass
class ZoneCondition:
    """A condition that evaluates based on detection count within a zone.

    Attributes:
        zone: The ``PolygonZone`` to evaluate.
        operator: Comparison operator (e.g. ``">="``, ``"=="``, ``"<"``).
        count_threshold: The count to compare against.
        description: Human-readable description of the condition.
    """

    zone: PolygonZone
    operator: str = ">="
    count_threshold: int = 1
    description: str = ""

    def evaluate(self, detections: Detections) -> bool:
        """Trigger the zone and evaluate the count condition.

        Args:
            detections: Detections container for the current frame.

        Returns:
            ``True`` if the zone count satisfies the condition.
        """
        self.zone.trigger(detections)
        count = self.zone.current_count

        op_enum = _STR_TO_OP.get(self.operator)
        if op_enum is None:
            logger.warning("Unknown operator %r, defaulting to >=", self.operator)
            op_enum = ComparisonOperator.GE

        result = cast(bool, _OP_FNS[op_enum](count, self.count_threshold))
        logger.debug(
            "ZoneCondition: %d %s %d → %s (%s)",
            count,
            self.operator,
            self.count_threshold,
            result,
            self.description or "unnamed",
        )
        return result


@dataclass
class ZoneConditionEvaluator:
    """Registry and evaluator for named zone conditions.

    Manages multiple ``ZoneCondition`` instances keyed by name,
    and evaluates them all against a ``Detections`` frame.

    Attributes:
        conditions: Mapping of condition name to ZoneCondition.
    """

    conditions: dict[str, ZoneCondition] = field(default_factory=dict)

    def register(self, name: str, condition: ZoneCondition) -> None:
        """Register a named zone condition.

        Args:
            name: Unique identifier for this condition (e.g. ``"inventory_full"``).
            condition: The zone condition to register.
        """
        self.conditions[name] = condition

    def unregister(self, name: str) -> None:
        """Remove a named zone condition."""
        self.conditions.pop(name, None)

    def evaluate(self, name: str, detections: Detections) -> bool:
        """Evaluate a single named condition.

        Args:
            name: The condition name.
            detections: Current-frame detections.

        Returns:
            ``True`` if the condition is satisfied.

        Raises:
            KeyError: If the condition name is not registered.
        """
        return self.conditions[name].evaluate(detections)

    def evaluate_all(self, detections: Detections) -> dict[str, bool]:
        """Evaluate all registered conditions.

        Args:
            detections: Current-frame detections.

        Returns:
            Dict mapping condition name to result.
        """
        return {name: cond.evaluate(detections) for name, cond in self.conditions.items()}

    def triggered(self, detections: Detections) -> list[str]:
        """Return names of conditions that are currently triggered.

        Args:
            detections: Current-frame detections.

        Returns:
            List of condition names where the condition evaluated to True.
        """
        return [name for name, result in self.evaluate_all(detections).items() if result]
