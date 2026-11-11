"""The questions stage of webshaper_frames: merge the sources into one
record format, drop duplicate questions and the fixed question types
(``filters.DROP_FRAMES_REASONING_TYPES``): ``questions.jsonl``, written again
by every build from the pinned sources (the same questions in the same
order).
"""

import logging
import re
from typing import List, Tuple

from utils.io_utils import write_jsonl

from ..config import DatasetPaths
from ..sources import LOADERS
from ..stats import drop_stats, question_stats, write_dropped, write_stats
from .filters import type_filter

log = logging.getLogger(__name__)


def _question_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _merge() -> Tuple[List[dict], List[dict]]:
    """The questions of every source, the duplicates dropped (the first
    source of a question keeps it)."""
    records, dropped, seen = [], [], {}
    for source, load in LOADERS.items():
        n = 0
        for record in load():
            key = _question_key(record["question"])
            if key in seen:
                dropped.append({"id": record["id"], "source": source, "reasons": ["duplicate"],
                                "duplicate_of": seen[key]})
                continue
            seen[key] = record["id"]
            records.append(record)
            n += 1
        log.info("questions: %s, %d questions", source, n)
    return records, dropped


def run_questions(paths: DatasetPaths) -> None:
    """The candidate questions (``questions.jsonl``), written again by every
    build from the pinned sources (the same questions in the same order)."""
    merged, duplicates = _merge()
    kept, by_type = type_filter(merged)
    write_jsonl(paths.questions, kept)
    write_dropped(paths.dropped, "questions", duplicates + by_type)
    write_stats(paths.stats, "questions", {"questions_merged": len(merged) + len(duplicates),
                                           "kept": question_stats(kept), "dropped": drop_stats(duplicates + by_type)})
    log.info("questions: kept %d of %d questions (%d duplicates, %d by type) -> %s",
             len(kept), len(merged) + len(duplicates), len(duplicates), len(by_type), paths.questions)
