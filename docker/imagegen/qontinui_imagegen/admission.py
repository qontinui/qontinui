"""Admission: one job on the card, at most one interactive request waiting (plan Phase 10).

Why the queue is this short. Every request the facade holds open is a request llama-swap
counts as in flight to the facade's process, and llama-swap will not swap that process out
while it has one: at the pinned v217 an exclusive swap calls ``Process.Stop()``, which waits
for in-flight requests (``proxy/process.go``), and a swap within a group waits in
``ProcessGroup.ProxyRequest`` behind the group lock. A queue inside the facade would
therefore hold the whole card -- ``paddleocr``, ``qontinui-grounding-v1`` and every other
alias would wait behind image work. So bulk work never waits here: it is refused with
``busy`` while a job runs, and ``deferred`` outside the bulk window, and queues in studio's
persistent scheduler instead. Only one interactive request may wait.

The interactive GPU-seconds cap bounds COMPUTE, not a cold start: its clock starts when the
engine reports the job's pipeline loaded and its weights verified (``Ticket.begin_compute``),
because Phase 0 measured a cold FP8 Z-Image-Turbo load at about 214-220 s on spaceship --
a cap that counted the load would fail every cold interactive request. So a job's wall time
is (load, when its pipeline is not resident) + (at most the cap), and a waiting request
waits for that much on top of its own. A client must size its timeout for a cold load plus
the cap; the runner's 300 s vision-client timeout covers a warm job, not a cold one queued
behind another cold one. At the pinned llama-swap v217 a further wait sits OUTSIDE the
facade: the first request after llama-swap swaps the facade in is proxied while the
process group's lock is held (``ProcessGroup.ProxyRequest``), so a second request for any
image alias waits inside llama-swap until that first request ends and never reaches the
facade to get a typed ``busy``. A client therefore sees, during a cold first request, a
slow answer rather than a fast refusal, and needs the same cold-load-plus-cap timeout.

The bulk window is read on an explicit clock: ``bulk_window.timezone`` in facade.yaml, or
``QONTINUI_IMAGEGEN_TIMEZONE``. A container's own clock is UTC unless told otherwise, and
"00:00-06:00" means the box's night, so with no zone configured the window is closed
(``deferred``, code ``bulk_window_timezone_unset``) rather than opened on a guessed zone. A
presence signal would also open it (D1.3), but none is readable from the facade yet
(Phase 0 item 10), and an unreadable presence signal means the window is closed (PD10).

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
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from qontinui_imagegen import errors
from qontinui_imagegen.config import Admission


@dataclass
class Ticket:
    """One admitted request. ``cancelled`` is set when its client goes away."""

    priority: str
    cap_s: float | None
    queued: bool = False
    #: When the ticket took the card (monotonic seconds); None while it waits.
    started_at: float | None = None
    #: When its compute began -- pipeline loaded, weights verified. The cap counts from here.
    compute_started_at: float | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    turn: asyncio.Event | None = None

    def begin_compute(self, now: float) -> None:
        if self.compute_started_at is None:
            self.compute_started_at = now

    def deadline(self) -> float | None:
        if self.cap_s is None or self.compute_started_at is None:
            return None
        return self.compute_started_at + self.cap_s


def zone_clock(zone: str | None) -> Callable[[], datetime | None]:
    """The box clock in ``zone`` (naive local time), or None-returning when no zone is set."""
    if zone is None:
        return lambda: None
    info = ZoneInfo(zone)
    return lambda: datetime.now(timezone.utc).astimezone(info).replace(tzinfo=None)


class AdmissionController:
    def __init__(
        self,
        settings: Admission,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime | None] | None = None,
    ) -> None:
        self.settings = settings
        self.monotonic = monotonic
        self.now = now or zone_clock(settings.bulk_window_timezone)
        self._running: Ticket | None = None
        self._waiter: Ticket | None = None

    # --- the bulk window ------------------------------------------------------------------

    def in_bulk_window(self, moment: datetime | None = None) -> bool:
        current = moment or self.now()
        if current is None:
            return False  # no zone configured: the box's night is unknown, so closed
        clock = current.time()
        start, end = self.settings.bulk_window_start, self.settings.bulk_window_end
        if start < end:
            return start <= clock < end
        return clock >= start or clock < end  # a window across midnight

    def seconds_until_window(self, moment: datetime | None = None) -> float | None:
        current = moment or self.now()
        if current is None:
            return None
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
        if priority == "bulk" and self.now() is None:
            raise errors.FacadeError(
                429,
                "deferred",
                "bulk_window_timezone_unset",
                "bulk work runs only in the bulk window, read on the box's clock, and no "
                "timezone is configured (facade.yaml bulk_window.timezone or "
                "QONTINUI_IMAGEGEN_TIMEZONE), so the window is closed",
            )
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
