"""Progress bars of the run stages (generate, belief, eval).

On a terminal the bars redraw in place.  Off a terminal (an sbatch ``.out``
file) every redraw would be appended verbatim, so the bars are off there and
each finished item prints one line instead:

* :class:`WorkerBars`: one bar per worker (generate: per GPU worker or the
  one sequential worker; belief: per belief thread), the postfix naming the
  worker's current query and step; off a terminal one line per finished
  query.
* :class:`StepBar` / :func:`step_iter`: one bar per eval step (loading the
  results, the judge calls); off a terminal one line when the step ends.
"""

import sys
import threading
import time
from typing import Iterable, Iterator, List, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")

_WORKER_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}{postfix}]"
_STEP_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]"


def live_bars() -> bool:
    """Whether bars redraw in place: stdout is a terminal."""
    return sys.stdout.isatty()


def _elapsed(start: float) -> str:
    return tqdm.format_interval(time.time() - start)


class WorkerBars:
    """One bar per worker, ``[{label} i]``, with the worker's current query
    and step as its postfix.  Thread-safe.

    Args:
        totals: The number of queries of each worker.
        label: The bar name ("Worker", "Belief").
        live: Redraw in place; None: :func:`live_bars`.  Off, each finished
            query prints one line.
    """

    def __init__(self, totals: List[int], label: str, live: Optional[bool] = None):
        self.label = label
        self.live = live_bars() if live is None else live
        self.totals = list(totals)
        self.done_counts = [0] * len(totals)
        self._lock = threading.Lock()
        self.bars = [
            tqdm(total=t, position=i, leave=True, desc=f"[{label} {i}]", bar_format=_WORKER_FORMAT,
                 dynamic_ncols=True, disable=not self.live)
            for i, t in enumerate(self.totals)
        ]
        if self.live:
            for bar in self.bars:
                bar.set_postfix_str("init", refresh=True)

    def status(self, worker: int, text: str) -> None:
        """The worker's current query and step, e.g. ``q12, turn 3/20, search``."""
        if not self.live:
            return
        with self._lock:
            self.bars[worker].set_postfix_str(text, refresh=True)

    def done(self, worker: int, qid: str, text: str) -> None:
        """The worker finished query *qid* (saved or not, *text* says)."""
        with self._lock:
            self.done_counts[worker] += 1
            if self.live:
                bar = self.bars[worker]
                bar.update(1)
                bar.set_postfix_str(f"{qid}, {text}", refresh=True)
            else:
                print(f"[{self.label} {worker}] {self.done_counts[worker]}/{self.totals[worker]} done "
                      f"({qid}, {text})", flush=True)

    def write(self, message: str) -> None:
        """A line above the bars (without scrambling them)."""
        with self._lock:
            if self.live:
                tqdm.write(message)
            else:
                print(message, flush=True)

    def close(self) -> None:
        for bar in self.bars:
            bar.close()


class StepBar:
    """One bar for one eval step; off a terminal one line when it closes."""

    def __init__(self, total: int, desc: str, unit: str = "it"):
        self.desc, self.total, self.n = desc, total, 0
        self.live = live_bars()
        self.start = time.time()
        self.bar = tqdm(total=total, desc=desc, unit=unit, bar_format=_STEP_FORMAT,
                        dynamic_ncols=True, disable=not self.live)

    def update(self, n: int = 1) -> None:
        self.n += n
        self.bar.update(n)

    def close(self) -> None:
        self.bar.close()
        if not self.live:
            print(f"{self.desc} {self.n}/{self.total} done [{_elapsed(self.start)}]", flush=True)

    def __enter__(self) -> "StepBar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def step_iter(iterable: Iterable[T], total: int, desc: str, unit: str = "it") -> Iterator[T]:
    """*iterable* under a :class:`StepBar`."""
    with StepBar(total, desc, unit) as bar:
        for item in iterable:
            yield item
            bar.update()
