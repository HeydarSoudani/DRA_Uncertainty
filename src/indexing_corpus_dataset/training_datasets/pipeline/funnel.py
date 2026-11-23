"""The filter funnel of stage data_prepare: each check of a dataset
applied in order, a record dropped at its first failed check."""

from collections import Counter
from typing import Callable, List, Optional

# A check: the reasons a record fails (``[{"status": code, ...}]``), or
# None when it passes.
Check = Callable[[dict], Optional[List[dict]]]


class Funnel:
    """The records still kept, the dropped ones (each at its first failed
    check) and the pending ones (a check that cannot decide yet, e.g. a web
    page still to fetch: neither kept nor dropped); per check, the records
    in and the drops per reason (``steps``)."""

    def __init__(self, records: List[dict]):
        self.kept: List[dict] = list(records)
        self.dropped: List[dict] = []
        self.pending: List[str] = []
        self.steps: List[dict] = []

    def apply(self, name: str, check: Check) -> None:
        kept, reasons_of = [], Counter()
        n_in = len(self.kept)
        for record in self.kept:
            reasons = check(record)
            if reasons:
                self.dropped.append({"id": record["id"], "source": record["source"], "filter": name,
                                     "reasons": reasons})
                reasons_of.update({r["status"] for r in reasons})
            else:
                kept.append(record)
        self.kept = kept
        self.steps.append({"filter": name, "in": n_in, "dropped": dict(reasons_of.most_common())})

    def hold(self, name: str, ids: set) -> None:
        """Set the records of *ids* aside as pending."""
        n_in = len(self.kept)
        self.pending += [r["id"] for r in self.kept if r["id"] in ids]
        self.kept = [r for r in self.kept if r["id"] not in ids]
        self.steps.append({"filter": name, "in": n_in, "dropped": {}, "pending": n_in - len(self.kept)})

    def table(self) -> str:
        """The funnel as plain text: one line per check."""
        width = max((len(s["filter"]) for s in self.steps), default=0)
        lines = []
        for s in self.steps:
            dropped = ", ".join(f"{code} {n}" for code, n in s["dropped"].items()) or "-"
            pending = f", pending {s['pending']}" if s.get("pending") else ""
            lines.append(f"  {s['filter'].ljust(width)}  in {s['in']:>6}  dropped: {dropped}{pending}")
        lines.append(f"  {'kept'.ljust(width)}  {len(self.kept):>9}")
        return "\n".join(lines)
