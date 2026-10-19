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

The interactive GPU-seconds cap bounds COMPUTE, never a load: its clock starts when the
engine reports everything the job loads loaded and its weights verified
(``Ticket.begin_compute``). Two loads sit before that point. A cold start loads the pipeline
(Phase 0 measured a cold FP8 Z-Image-Turbo load at about 214-220 s on spaceship -- a cap
that counted it would fail every cold interactive request), and under the
``sequential_text_encoder`` placement every job, warm or cold, loads its text encoder (Phase
0: 52-55 s for Z-Image's), while under ``model_cpu_offload`` (Qwen) every job moves its text
encoder and then its transformer between host and card. The encode is compute: inside the
cap under the sequential placement, and outside it under offload, where it sits between the
two moves. So a job's wall time is its loads plus at most the cap, and a waiting request waits
for that much on top of its own. A client must size its timeout for the loads plus the cap;
the runner's 300 s vision-client timeout covers a warm job, not a cold one queued behind
another cold one. At the pinned llama-swap v217 a further wait sits OUTSIDE the facade: the
first request after llama-swap swaps the facade in is proxied while the process group's
lock is held (``ProcessGroup.ProxyRequest``), so a second request for any image alias waits
inside llama-swap until that first request ends and never reaches the facade to get a typed
``busy``. A client therefore sees, during a cold first request, a slow answer rather than a
fast refusal, and needs the same loads-plus-cap timeout.

When a running job should end is therefore its remaining expected load (``Ticket.begin_load``,
from facade.yaml's ``load_estimate``) plus the cap while it loads, and its cap deadline once
it computes. A bulk job has no cap, and a load that overran its estimate has no honest
remaining time, so both report their end as unknown (``None``); ``busy``'s ``Retry-After``
then falls back to one cap, stated as a re-ask interval (``retry_after_basis``), never as an
end estimate.

The bulk window is read on an explicit clock: ``bulk_window.timezone`` in facade.yaml, or
``QONTINUI_IMAGEGEN_TIMEZONE``. A container's own clock is UTC unless told otherwise, and
"00:00-06:00" means the box's night, so with no zone configured the window is closed
(``deferred``, code ``bulk_window_timezone_unset``) rather than opened on a guessed zone. A
presence signal would also open it (D1.3), but none is readable from the facade yet
(Phase 0 item 10), and an unreadable presence signal means the window is closed (PD10).
An unset zone is a per-request refusal, not a startup error: interactive work needs no
window and runs any time (D1.3), and a facade that refused to start would take every image
alias down over a setting only bulk work reads. Its retry time is the window's own length:
when no clock says when the window opens, a client that re-asks at least that often cannot
step over a whole window once the zone is set.

All of the controller's own state changes happen on the event loop's thread with no
``await`` between a check and its change, so it needs no lock; a finishing job hands the
card straight to the waiter, so nothing can slip in between. The card's thread writes only
a running ticket's timestamps (``begin_load``, ``begin_compute``), each set once, and the
event loop reads them as they stand.
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
    #: When its job began on the card (weight checks, then its loads), and how long those
    #: loads are expected to take (facade.yaml load_estimate); None until the job begins.
    load_started_at: float | None = None
    expected_load_s: float | None = None
    #: When its compute began -- everything loaded, weights verified. The cap counts from here.
    compute_started_at: float | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    turn: asyncio.Event | None = None

    def begin_load(self, now: float, expected_load_s: float) -> None:
        if self.load_started_at is None:
            # The estimate first: the event loop reads these without a lock, and a set
            # load_started_at is what tells it the estimate is there.
            self.expected_load_s = expected_load_s
            self.load_started_at = now

    def begin_compute(self, now: float) -> None:
        if self.compute_started_at is None:
            self.compute_started_at = now

    @property
    def phase(self) -> str:
        if self.compute_started_at is not None:
            return "computing"
        if self.load_started_at is not None:
            return "loading"
        return "waiting" if self.started_at is None else "starting"

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

    def window_length_s(self) -> float:
        start, end = self.settings.bulk_window_start, self.settings.bulk_window_end
        minutes = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
        return float((minutes % (24 * 60)) * 60)

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

    def _remaining(self, ticket: Ticket) -> float | None:
        """Seconds until ``ticket``'s job should end, or None when no estimate holds.

        Computing: until its cap deadline. Loading: the rest of its expected load, then a
        whole cap. Unknown: a bulk job (no cap), a job not yet begun on the card, or a load
        past its estimate (how much longer it takes is not known).
        """
        now = self.monotonic()
        deadline = ticket.deadline()
        if deadline is not None:
            return max(1.0, deadline - now)
        started, expected = ticket.load_started_at, ticket.expected_load_s
        if ticket.cap_s is None or started is None or expected is None:
            return None
        load_left = started + expected - now
        if load_left <= 0:
            return None
        return load_left + ticket.cap_s

    def _retry_after(self) -> tuple[float, str]:
        """When to re-ask while the card is busy, and what that number is based on."""
        cap = self.settings.interactive_gpu_seconds_cap
        remaining = None if self._running is None else self._remaining(self._running)
        basis = "estimated_end"
        if remaining is None:
            remaining, basis = cap, "reask_interval"  # the running job's end is unknown
        # A waiter runs next: at least one more cap (its own loads are not known here).
        waiting = cap if self._waiter is not None else 0.0
        return remaining + waiting, basis

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
            raise errors.deferred(
                "bulk_window_timezone_unset",
                "bulk work runs only in the bulk window, read on the box's clock, and no "
                "timezone is configured (facade.yaml bulk_window.timezone or "
                "QONTINUI_IMAGEGEN_TIMEZONE), so the window is closed; retry_after_s is the "
                "window's length, so a client re-asking that often cannot miss a window "
                "once the zone is set",
                self.window_length_s(),
            )
        if priority == "bulk" and not self.in_bulk_window():
            until = self.seconds_until_window()
            assert until is not None  # a zone is set: now() answered above
            raise errors.deferred(
                "outside_bulk_window",
                "bulk work runs only in the bulk window "
                f"({self.settings.bulk_window_start:%H:%M}-{self.settings.bulk_window_end:%H:%M} "
                "box time) while no presence signal is readable",
                until,
            )
        if self._running is None and self._waiter is None:
            return self._start(Ticket(priority, cap))
        if (
            priority == "bulk"
            or self._waiter is not None
            or self.settings.max_interactive_waiters == 0
        ):
            retry_after, basis = self._retry_after()
            raise errors.busy(
                "card_busy",
                "an image job is running"
                + (" and another is waiting" if self._waiter is not None else "")
                + "; bulk work queues in its scheduler, not here",
                retry_after,
                retry_after_basis=basis,
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
        """The card as read now. ``None`` for a time means unknown, never zero."""
        running = self._running
        if running is None:
            card = None
        else:
            deadline = running.deadline()
            card = {
                "priority": running.priority,
                "phase": running.phase,
                # The cap's own deadline: None until compute begins (the cap never counts a
                # load), and for bulk work, which has no cap.
                "deadline_in_s": None
                if deadline is None
                else max(0.0, deadline - self.monotonic()),
                # When the job should end, loads included; None when no estimate holds.
                "expected_end_in_s": self._remaining(running),
            }
        return {
            "running": card,
            "waiting": self._waiter is not None,
            "bulk_window_open": self.in_bulk_window(),
            "bulk_window_timezone": self.settings.bulk_window_timezone,
        }
