"""Admission: one job on the card, at most one interactive request waiting (plan Phase 10).

Why the queue is this short. Every request the facade holds open is a request llama-swap
counts as in flight to the facade's process, and llama-swap will not swap that process out
while it has one: at the pinned v217 an exclusive swap calls ``Process.Stop()``, which waits
for in-flight requests (``proxy/process.go``), and a swap within a group waits in
``ProcessGroup.ProxyRequest`` behind the group lock. A queue inside the facade would
therefore hold the whole card -- ``paddleocr``, ``qontinui-grounding-v1`` and every other
alias would wait behind image work. So bulk work never waits here: it is refused with
``busy`` while a job runs, and ``deferred`` outside the bulk window, and queues in studio's
persistent scheduler instead. Only one interactive request may wait, which keeps wait + job
under twice the interactive GPU-seconds cap -- inside the runner's 300 s vision timeout.

The bulk window is read on the box clock. A presence signal would also open it (D1.3), but
none is readable from the facade yet (Phase 0 item 10), and an unreadable presence signal
means the window is closed (PD10).

All state changes happen on the event loop's thread with no ``await`` between a check and
its change, so the controller needs no lock; a finishing job hands the card straight to the
waiter, so nothing can slip in between.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from qontinui_imagegen import errors
from qontinui_imagegen.config import Admission


@dataclass
class Ticket:
    """One admitted request. ``cancelled`` is set when its client goes away."""

    priority: str
    cap_s: float | None
    queued: bool = False
    started_at: float | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    turn: asyncio.Event | None = None

    def deadline(self) -> float | None:
        if self.cap_s is None or self.started_at is None:
            return None
        return self.started_at + self.cap_s


class AdmissionController:
    def __init__(
        self,
        settings: Admission,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.settings = settings
        self.monotonic = monotonic
        self.now = now
        self._running: Ticket | None = None
        self._waiter: Ticket | None = None

    # --- the bulk window ------------------------------------------------------------------

    def in_bulk_window(self, moment: datetime | None = None) -> bool:
        clock = (moment or self.now()).time()
        start, end = self.settings.bulk_window_start, self.settings.bulk_window_end
        if start < end:
            return start <= clock < end
        return clock >= start or clock < end  # a window across midnight

    def seconds_until_window(self, moment: datetime | None = None) -> float:
        current = moment or self.now()
        opening = current.replace(
            hour=self.settings.bulk_window_start.hour,
            minute=self.settings.bulk_window_start.minute,
            second=0,
            microsecond=0,
        )
        if opening <= current:
            opening += timedelta(days=1)
        return (opening - current).total_seconds()

    # --- the card --------------------------------------------------------------------------

    def _remaining(self, ticket: Ticket | None) -> float:
        """Seconds until ``ticket``'s job must end, or a whole cap when that is unknown."""
        if ticket is None:
            return 0.0
        deadline = ticket.deadline()
        if deadline is None:
            return self.settings.interactive_gpu_seconds_cap
        return max(1.0, deadline - self.monotonic())

    def _retry_after(self) -> float:
        waiting = self.settings.interactive_gpu_seconds_cap if self._waiter is not None else 0.0
        return self._remaining(self._running) + waiting

    def _start(self, ticket: Ticket) -> Ticket:
        ticket.started_at = self.monotonic()
        self._running = ticket
        return ticket

    def admit(self, priority: str) -> Ticket:
        """Take the card now, take the one waiting place, or refuse with a typed answer.

        A ticket with ``queued`` set must be awaited with :meth:`wait_turn` before its job
        runs, and every ticket must reach :meth:`release` (it ran) or :meth:`abandon` (it
        never did).
        """
        cap = self.settings.interactive_gpu_seconds_cap if priority == "interactive" else None
        if priority == "bulk" and not self.in_bulk_window():
            raise errors.deferred(
                "bulk work runs only in the bulk window "
                f"({self.settings.bulk_window_start:%H:%M}-{self.settings.bulk_window_end:%H:%M} "
                "box time) while no presence signal is readable",
                self.seconds_until_window(),
            )
        if self._running is None and self._waiter is None:
            return self._start(Ticket(priority, cap))
        if (
            priority == "bulk"
            or self._waiter is not None
            or self.settings.max_interactive_waiters == 0
        ):
            raise errors.busy(
                "card_busy",
                "an image job is running"
                + (" and another is waiting" if self._waiter is not None else "")
                + "; bulk work queues in its scheduler, not here",
                self._retry_after(),
            )
        ticket = Ticket(priority, cap, queued=True, turn=asyncio.Event())
        self._waiter = ticket
        return ticket

    async def wait_turn(self, ticket: Ticket) -> None:
        if ticket.turn is not None:
            await ticket.turn.wait()

    def abandon(self, ticket: Ticket) -> None:
        """A ticket that will never run gives back its waiting place."""
        if self._waiter is ticket:
            self._waiter = None

    def release(self, ticket: Ticket) -> None:
        """The ticket's job ended: hand the card to the waiter, or free it."""
        if self._running is not ticket:
            return
        self._running = None
        waiter, self._waiter = self._waiter, None
        if waiter is not None:
            self._start(waiter)
            assert waiter.turn is not None
            waiter.turn.set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": None
            if self._running is None
            else {
                "priority": self._running.priority,
                "deadline_in_s": self._remaining(self._running),
            },
            "waiting": self._waiter is not None,
            "bulk_window_open": self.in_bulk_window(),
        }
