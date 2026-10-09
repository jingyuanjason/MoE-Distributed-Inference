"""Core load generation engine.

Slices the requested per-second rate into 10 ms dispatch ticks (see
:class:`TickSchedule`), samples conversations from a
:class:`prompt_sampler.PromptSampler`, and fans them out to an
OpenAI-compatible chat-completions endpoint as individual asyncio tasks.
Multi-round conversations are replayed one user message at a time, waiting
for the assistant reply before appending the next message.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

from core.prompt_sampler import PromptSampler

#: Dispatch period in seconds: every 10 ms a batch of requests is released.
TICK_INTERVAL_SECONDS: float = 0.01

#: How many times a single message exchange is attempted before the whole
#: request is marked as failed.
MAX_RETRIES: int = 3


class LoadAlreadyRunningError(RuntimeError):
    """Raised when a load run is started while another run is active."""


@dataclass
class GeneratorSettings:
    """Static generator behavior settings, loaded from the YAML config file.

    These describe *how* the load generator behaves (retries, timeouts,
    dispatch granularity). Everything about *where* requests go comes from
    the per-run :class:`LoadConfig` supplied via the API.
    """

    max_retries: int = MAX_RETRIES
    retry_backoff_base_seconds: float = 0.5
    request_timeout_seconds: float = 60.0
    tick_interval_seconds: float = TICK_INTERVAL_SECONDS

    @classmethod
    def from_file(cls, config_path: str | Path) -> "GeneratorSettings":
        """Load behavior settings from a YAML configuration file.

        Missing sections or keys fall back to the defaults.
        """
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        behavior = raw.get("behavior", {})
        dispatch = raw.get("dispatch", {})
        return cls(
            max_retries=int(behavior.get("max_retries", cls.max_retries)),
            retry_backoff_base_seconds=float(
                behavior.get(
                    "retry_backoff_base_seconds", cls.retry_backoff_base_seconds
                )
            ),
            request_timeout_seconds=float(
                behavior.get("request_timeout_seconds", cls.request_timeout_seconds)
            ),
            tick_interval_seconds=float(
                dispatch.get("tick_interval_seconds", cls.tick_interval_seconds)
            ),
        )


@dataclass
class LoadConfig:
    """Configuration for a single load generation run.

    Carries the full endpoint information (from the API request) plus the
    desired rate and duration.
    """

    ip_address: str
    samples_per_second: float
    duration_seconds: float
    port: int = 8000
    model: str = "default"

    @property
    def base_url(self) -> str:
        return f"http://{self.ip_address}:{self.port}"


@dataclass
class TickSchedule:
    """Pre-computed batching plan derived from a :class:`LoadConfig`.

    The overall per-second rate is sliced into 10 ms ticks. Each tick
    releases ``requests_per_tick[i]`` requests, where the fractional part of
    the rate is spread evenly across the ticks so the long-term average
    matches ``samples_per_second`` exactly.
    """

    tick_interval_seconds: float
    ticks_total: int
    requests_per_tick: list[int]

    @property
    def total_requests(self) -> int:
        return sum(self.requests_per_tick)

    @classmethod
    def from_config(
        cls, config: LoadConfig, tick_interval_seconds: float = TICK_INTERVAL_SECONDS
    ) -> "TickSchedule":
        """Slice ``config.samples_per_second`` into per-tick batch sizes.

        The mean number of requests per tick is ``rate * interval``; the
        integer floor is issued every tick and the fractional remainder is
        accumulated so that one extra request is issued whenever the
        accumulator rolls over a whole request (Bresenham-style), keeping
        the average rate exact while batches stay as uniform as possible.
        """
        ticks_total = max(1, math.ceil(config.duration_seconds / tick_interval_seconds))
        mean_per_tick = config.samples_per_second * tick_interval_seconds
        base = math.floor(mean_per_tick)
        remainder = mean_per_tick - base

        requests_per_tick: list[int] = []
        accumulator = 0.0
        for _ in range(ticks_total):
            accumulator += remainder
            # The epsilon absorbs float accumulation error (e.g. 0.1 added
            # ten times yields 0.9999999999999999 instead of 1.0).
            extra = math.floor(accumulator + 1e-9)
            accumulator -= extra
            requests_per_tick.append(base + extra)

        return cls(
            tick_interval_seconds=tick_interval_seconds,
            ticks_total=ticks_total,
            requests_per_tick=requests_per_tick,
        )


@dataclass
class RequestResult:
    """Outcome of one sampled request (possibly multi-round)."""

    source: str
    rounds_total: int
    rounds_completed: int
    success: bool
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_seconds: float = 0.0  # sum of successful round latencies

    @property
    def duration_seconds(self) -> float:
        """End-to-end wall-clock time of the request, including retries."""
        return self.finished_at - self.started_at

    @property
    def tokens_per_second(self) -> float:
        """Completion-token throughput while the endpoint was generating."""
        if self.generation_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.generation_seconds


@dataclass
class LoadRun:
    """Represents an active (or scheduled) load generation run."""

    config: LoadConfig
    schedule: TickSchedule
    task: asyncio.Task | None = field(default=None, repr=False)
    results: list[RequestResult] = field(default_factory=list, repr=False)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def completed_requests(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_requests(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def stats(self) -> dict[str, float | int | None]:
        """Aggregate statistics for the run.

        Rates are computed over finished requests so far; ``duration_seconds``
        is the wall-clock time from run start until it finished (or now, if
        still running). Token throughputs are split by direction:
        ``avg_input_tokens_per_second`` / ``avg_output_tokens_per_second``
        average the per-request rates (tokens / generation time), while
        ``input_tokens_per_second`` / ``output_tokens_per_second`` divide the
        run totals by the run duration.
        """
        total = len(self.results)
        failed = self.failed_requests
        finished = self.finished_at if self.finished_at is not None else time.time()
        duration = finished - self.started_at
        prompt_tokens = sum(r.prompt_tokens for r in self.results)
        completion_tokens = sum(r.completion_tokens for r in self.results)

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        successful = [r for r in self.results if r.success]
        return {
            "duration_seconds": duration,
            "requests_total": total,
            "requests_succeeded": total - failed,
            "requests_failed": failed,
            "success_rate": (total - failed) / total if total else 0.0,
            "failure_rate": failed / total if total else 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "avg_input_tokens_per_second": mean(
                [
                    r.prompt_tokens / r.generation_seconds
                    for r in successful
                    if r.generation_seconds > 0
                ]
            ),
            "avg_output_tokens_per_second": mean(
                [r.tokens_per_second for r in successful]
            ),
            "input_tokens_per_second": prompt_tokens / duration if duration > 0 else 0.0,
            "output_tokens_per_second": (
                completion_tokens / duration if duration > 0 else 0.0
            ),
        }


class LoadGenerator:
    """Highly parallel load generator for LLM serving workloads.

    On :meth:`start_load`, the requested rate is sliced into per-tick batch
    sizes (see :class:`TickSchedule`). A background dispatch task then, every
    ``tick_interval_seconds``, samples that many conversations from the
    :class:`PromptSampler` and dispatches each one as its own asyncio task.
    Multi-round conversations are replayed one user message at a time: each
    message is sent to the endpoint, the assistant reply is awaited and
    appended to the conversation history before the next message is sent.
    A message exchange is retried up to ``max_retries`` times with
    exponential backoff; if it still fails, the whole request is recorded
    with a failed marker.
    """

    def __init__(self, sampler: PromptSampler, settings: GeneratorSettings) -> None:
        self._sampler = sampler
        self._settings = settings
        self._runs: list[LoadRun] = []
        # Guards against overlapping runs: held for the whole lifetime of
        # an active run and released when the run's task completes.
        self._run_lock = asyncio.Lock()

    async def start_load(self, config: LoadConfig) -> LoadRun:
        """Start generating load toward the given target.

        Parses the requested rate into a :class:`TickSchedule` that slices
        the per-second rate into per-10 ms request batches, then launches
        the background dispatch task.

        Raises:
            LoadAlreadyRunningError: if another load run is still active.
        """
        if self._run_lock.locked():
            raise LoadAlreadyRunningError(
                "a load generation run is already in progress; "
                "wait for it to finish before starting a new one"
            )
        await self._run_lock.acquire()

        schedule = TickSchedule.from_config(
            config, tick_interval_seconds=self._settings.tick_interval_seconds
        )
        run = LoadRun(config=config, schedule=schedule)
        run.task = asyncio.create_task(self._dispatch(run))
        run.task.add_done_callback(self._on_run_finished)
        self._runs.append(run)
        return run

    def _on_run_finished(self, task: asyncio.Task) -> None:
        """Stamp the run's finish time and release the lock for new runs."""
        for run in self._runs:
            if run.task is task:
                run.finished_at = time.time()
                break
        self._release_run_lock()

    async def _dispatch(self, run: LoadRun) -> None:
        """Release request batches tick by tick, then await completion."""
        base_url = run.config.base_url
        timeout = httpx.Timeout(self._settings.request_timeout_seconds)
        pending: list[asyncio.Task] = []
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            loop = asyncio.get_running_loop()
            start = loop.time()
            for tick_index, batch_size in enumerate(run.schedule.requests_per_tick):
                for _ in range(batch_size):
                    queries, source = self._sampler.sample_with_source()
                    pending.append(
                        asyncio.create_task(
                            self._run_conversation(client, run, queries, source)
                        )
                    )
                # Sleep until the next tick boundary (don't drift).
                next_tick = start + (tick_index + 1) * run.schedule.tick_interval_seconds
                await asyncio.sleep(max(0.0, next_tick - loop.time()))
            if pending:
                await asyncio.gather(*pending)

    async def _run_conversation(
        self,
        client: httpx.AsyncClient,
        run: LoadRun,
        queries: list[str],
        source: str,
    ) -> None:
        """Replay one sampled conversation against the endpoint.

        Sends one message at a time, appending each assistant reply to the
        history before sending the next user message. On failure after all
        retries, records a failed marker and stops the conversation.
        """
        started_at = time.time()
        messages: list[dict[str, str]] = []
        rounds_completed = 0
        prompt_tokens = 0
        completion_tokens = 0
        generation_seconds = 0.0
        error: str | None = None

        for query in queries:
            messages.append({"role": "user", "content": query})
            round_start = time.monotonic()
            reply, usage, error = await self._send_with_retries(client, run, messages)
            if reply is None:
                break
            generation_seconds += time.monotonic() - round_start
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            messages.append({"role": "assistant", "content": reply})
            rounds_completed += 1

        run.results.append(
            RequestResult(
                source=source,
                rounds_total=len(queries),
                rounds_completed=rounds_completed,
                success=error is None,
                error=error,
                started_at=started_at,
                finished_at=time.time(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                generation_seconds=generation_seconds,
            )
        )

    async def _send_with_retries(
        self,
        client: httpx.AsyncClient,
        run: LoadRun,
        messages: list[dict[str, str]],
    ) -> tuple[str | None, dict[str, int], str | None]:
        """POST one chat-completion request with exponential backoff retries.

        Up to ``max_retries`` attempts are made; between attempts the
        coroutine sleeps ``retry_backoff_base_seconds * 2 ** (attempt - 1)``.
        Returns ``(reply, usage, None)`` on success or ``(None, {}, error)``
        after all attempts failed. ``usage`` holds the token counts reported
        by the endpoint (empty dict if absent).
        """
        payload = {"model": run.config.model, "messages": messages}
        max_retries = self._settings.max_retries
        backoff_base = self._settings.retry_backoff_base_seconds
        last_error = "unknown error"
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                reply = body["choices"][0]["message"]["content"]
                usage = body.get("usage") or {}
                return reply, usage, None
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = f"attempt {attempt}/{max_retries}: {exc}"
                if attempt < max_retries:
                    await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
        return None, {}, last_error

    def _release_run_lock(self) -> None:
        """Release the run lock once the active run has finished."""
        if self._run_lock.locked():
            self._run_lock.release()

    @property
    def active_runs(self) -> list[LoadRun]:
        return list(self._runs)
