"""Progress of the LLM phases of stage data_relevance (the fact LLM, the
judge) and of the passage lookups.

On a terminal a bar redraws in place, its postfix the outcomes so far and
the tokens of the calls; off a terminal (an sbatch ``.out`` file) every
redraw would be appended, so one line is printed every ``LOG_EVERY``
seconds instead.  Either way a summary line ends the phase.
"""

import sys
import time
from collections import Counter
from typing import Any, Iterable, Optional

from tqdm import tqdm

# Seconds between two progress lines off a terminal.
LOG_EVERY = 30.0


def _tokens(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else str(n)


class CallBar:
    """The progress of *total* items of a phase (*desc*, e.g. ``[judge]``),
    each with an outcome (``update``); *meter*: the token meter of the
    phase's LLM client (``utils.token_meter.TokenMeter``), None without
    one."""

    def __init__(self, total: int, desc: str, meter: Any = None, unit: str = "call") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.meter = meter
        self.live = sys.stdout.isatty()
        self.counts: Counter = Counter()
        self.done = 0
        self._start = self._last = time.time()
        self._base = self._snapshot()
        self.bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, disable=not self.live,
                        mininterval=1.0)

    def _snapshot(self) -> Optional[dict]:
        return self.meter.snapshot() if self.meter is not None else None

    def _status(self) -> str:
        parts = [f"{k} {v}" for k, v in sorted(self.counts.items())]
        now = self._snapshot()
        if now is not None and self._base is not None:
            parts.append(f"tokens in {_tokens(now['input_tokens'] - self._base['input_tokens'])} "
                         f"out {_tokens(now['output_tokens'] - self._base['output_tokens'])}")
        return ", ".join(parts)

    def update(self, outcome: str, n: int = 1) -> None:
        """*n* more items done, with *outcome* (``ok``, ``failed``, ...)."""
        self.counts[outcome] += n
        self.done += n
        if self.live:
            self.bar.update(n)
            self.bar.set_postfix_str(self._status(), refresh=False)
        elif time.time() - self._last >= LOG_EVERY:
            self._last = time.time()
            print(f"{self.desc} {self.done}/{self.total} {self.unit}s "
                  f"({tqdm.format_interval(self._last - self._start)}): {self._status()}", flush=True)

    def close(self) -> None:
        self.bar.close()
        print(f"{self.desc} done: {self.done}/{self.total} {self.unit}s in "
              f"{tqdm.format_interval(time.time() - self._start)}" + (f"; {self._status()}" if self.counts else ""),
              flush=True)

    def __enter__(self) -> "CallBar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def batches(items: Iterable[Any], size: int):
    """*items* in lists of at most *size*."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
