"""Per-run measurements, bounded API work and explicit optional degradation."""
from __future__ import annotations

import time
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass


class BudgetExceeded(RuntimeError):
    """A configured limit was reached; never retry or degrade this failure."""


@dataclass(frozen=True)
class RuntimePolicy:
    deadline_seconds: float = 180.0
    api_timeout_seconds: float = 60.0
    max_api_calls: int = 8
    max_completion_tokens: int = 12000
    max_first_queries: int = 4
    max_followup_queries: int = 2
    max_retrieval_rounds: int = 2
    failure_policy: str = "strict"

    def __post_init__(self):
        if (not math.isfinite(self.deadline_seconds) or not math.isfinite(self.api_timeout_seconds)
                or self.deadline_seconds <= 0 or self.api_timeout_seconds <= 0):
            raise ValueError("运行时限和 API 超时必须大于 0")
        if self.max_api_calls < 1 or self.max_completion_tokens < 1:
            raise ValueError("API 调用/输出 token 预算必须大于 0")
        if not 1 <= self.max_first_queries <= 4 or not 0 <= self.max_followup_queries <= 2:
            raise ValueError("第一轮查询限额为 1–4，补查限额为 0–2")
        if self.max_retrieval_rounds not in {1, 2}:
            raise ValueError("检索轮数只能为 1 或 2")
        if self.failure_policy not in {"strict", "degrade"}:
            raise ValueError("failure_policy 必须是 strict 或 degrade")


_ACTIVE: ContextVar[RunRuntime | None] = ContextVar("litagent_runtime", default=None)


class RunRuntime:
    def __init__(self, policy: RuntimePolicy | None = None, *, clock=time.perf_counter):
        self.policy = policy or RuntimePolicy()
        self.clock = clock
        self.started = clock()
        self.timings: list[dict] = []
        self.degradations: list[dict] = []
        self.api_attempts = 0
        self.reserved_completion_tokens = 0
        self._stack: list[str] = []

    def remaining(self) -> float:
        return self.policy.deadline_seconds - (self.clock() - self.started)

    def check(self):
        if self.remaining() <= 0:
            raise BudgetExceeded("运行时间预算已耗尽")

    def claim_api(self, requested_timeout: float, max_tokens: int | None) -> float:
        self.check()
        reserve = max_tokens if max_tokens is not None else 2048
        if reserve < 1:
            raise ValueError("max_tokens 必须大于 0")
        if self.api_attempts >= self.policy.max_api_calls:
            raise BudgetExceeded("API 调用次数预算已耗尽")
        if self.reserved_completion_tokens + reserve > self.policy.max_completion_tokens:
            raise BudgetExceeded("API 最大输出 token 预算已耗尽")
        self.api_attempts += 1
        self.reserved_completion_tokens += reserve
        return min(float(requested_timeout), self.policy.api_timeout_seconds, self.remaining())

    @contextmanager
    def span(self, stage: str, **metadata):
        self.check()
        start = self.clock()
        depth = len(self._stack)
        parent = self._stack[-1] if self._stack else None
        self._stack.append(stage)
        status, error_type = "ok", None
        try:
            yield
            self.check()
        except BaseException as exc:
            status, error_type = "failed", type(exc).__name__
            raise
        finally:
            self._stack.pop()
            self.timings.append({"stage": stage, "seconds": round(self.clock() - start, 6),
                                 "status": status, "depth": depth, "parent_stage": parent,
                                 **metadata, **({"error_type": error_type} if error_type else {})})

    def degrade(self, stage: str, fallback: str, exc: Exception) -> bool:
        if (self.policy.failure_policy != "degrade" or isinstance(exc, BudgetExceeded)
                or not isinstance(exc, (RuntimeError, OSError))):
            return False
        self.check()
        self.degradations.append({"stage": stage, "fallback": fallback,
                                  "error_type": type(exc).__name__})
        return True

    def summary(self) -> dict:
        return {"policy": asdict(self.policy), "api_attempts": self.api_attempts,
                "reserved_completion_tokens": self.reserved_completion_tokens,
                "budget_semantics": "output caps reserved before requests; cooperative stage deadline plus socket timeout; local native calls cannot be preempted"}


def current_runtime() -> RunRuntime | None:
    return _ACTIVE.get()


@contextmanager
def use_runtime(runtime: RunRuntime):
    token = _ACTIVE.set(runtime)
    try:
        yield runtime
    finally:
        _ACTIVE.reset(token)


@contextmanager
def timed(stage: str, **metadata):
    runtime = current_runtime()
    if runtime is None:
        yield
    else:
        with runtime.span(stage, **metadata):
            yield


def allow_degradation(stage: str, fallback: str, exc: Exception) -> bool:
    runtime = current_runtime()
    return runtime.degrade(stage, fallback, exc) if runtime is not None else False
