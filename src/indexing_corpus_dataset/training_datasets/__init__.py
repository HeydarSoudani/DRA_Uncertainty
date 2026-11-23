"""Training datasets (infoseek_hard, frames, webshaper, trqa_val), each a
dataset of dra_inference, built in two stages
(``experiments/training/train_belief_model.py``):

    --stage data_prepare --dataset D     (``prepare``, no LLM call)
        the source's questions as loaded, nothing removed, written to
        data/training_datasets_raw/D (``DatasetPaths.raw_questions``, with
        the caches of the titles looked up and the web pages fetched);
        the questions that pass the dataset's checks (repeated question,
        answer, gold links and pages in the corpus; infoseek_hard: also
        the hardness cells, the tree, a named hop, the claim targets) to
        data/training_datasets_raw/D/prepared.jsonl, each with its
        features; the dropped ones with their filter and reasons to
        data/training_datasets_raw/D/dropped.jsonl
    --stage data_relevance --dataset D --limit N   (``build_relevance``, LLM calls)
        the gold annotation, qrels and split of the prepared questions,
        incremental: the next N not processed yet

The training datasets share one folder, data/training_datasets, laid out
like an inference dataset: every training dataset has one row per query in
queries/D.jsonl (the question, answer, source, split and gold annotation,
``annotation/rows.py``), its qrels in qrels/D.txt and its silver beliefs
in belief_bank/silver/D.jsonl.  The splits ``sft``, ``rl`` and ``val``
(``pipeline/splits.py``) are views of those files, read by
``dataset_loaders``.  Every build file (the LLM memories, the dropped
questions, stats.json) stays in data/training_datasets_raw/D.

Corpus (``corpus/``): corpus_wiki, InfoSeek's wiki-25-512 (Wikipedia 2025
in chunks of 512 tokens, downloaded once into DATA_ROOT/infoseek/corpus),
searched by infoseek_hard, frames and webshaper; corpus_web, WebShaper's
web gold pages cut the same way, in data/training_datasets/corpus (the
user merges the two for the index).  trqa_val searches TRQA's corpus.

Gold passages (``gold_passages/``): a gold page is found by its title;
the gold passages (chunks) of a page are found by an LLM judge, one call
per (fact, passage) pair, TRQA's property check.

infoseek_hard: InfoSeek-Hard questions of the hardness cells; the facts are
the claims of the research tree, each judged on chunk 2 of its page
(InfoSeek writes claims from the first 1024 tokens: chunk 2 when the judge
finds it there, else chunk 1); binary qrels, one belief per hop.

frames: FRAMES questions; one fact per gold page written by an LLM, judged
on every chunk of the page; graded qrels; no beliefs.

webshaper: WebShaper questions; one fact per triple of its formalization
(the variables named by an LLM), judged on every passage of the pages
assigned to it; graded qrels; one belief per variable.

trqa_val: the TRQA wiki1 + wiki2 validation questions with their gold
annotation from the intermediate info, TRQA's qrels, the belief texts and
the split sft / rl / val; it searches TRQA's corpus.

``stats.json`` holds one section per stage (``prepare`` with the funnel of
the checks, ``corpus``, ``web``; ``relevance``, ``gold_passages``,
``annotations``, ``split``).

Layout (each layer imports only the ones above it; nothing imports
``datasets/`` but ``registry``):

    config          settings (web, split, gold_beliefs, gold_passages) and
                    the paths of a dataset (raw folder, dataset files)
    registry        the datasets and their stage modules
    pipeline/       the shared steps: the source record format and
                    Hugging Face files (source), the funnel and the shared
                    checks (funnel, checks), the split (splits), stats.json
                    (stats); the stage drivers (prepare, relevance)
    corpus/         corpus_wiki (wiki, wiki_pages), corpus_web (web, the
                    fetchers in fetch/), passage texts (texts), gold page
                    entries (pages)
    annotation/     the query rows (rows), name matching (names), the
                    belief texts by an LLM (belief_texts, prompts/), the
                    ``annotations`` stats section (stats)
    gold_passages/  the fact writer (facts), the judge (judge), the gold
                    passages of frames and webshaper (select); prompts/
    datasets/       one package per dataset: source, prepare, relevance
                    (+ its facts, tree or formalization, fact prompts)

The TRQA gold beliefs of the belief model (made from TRQA's intermediate
info at inference) are in ``belief_model_component/init/trqa_gold.py``.
"""

from typing import Optional

from .config import BuildConfig, DatasetPaths, load_config

__all__ = ["BuildConfig", "DatasetPaths", "build_relevance", "load_config", "prepare"]


def prepare(cfg: BuildConfig, paths: DatasetPaths) -> None:
    """Stage data_prepare of the dataset of *paths* (``pipeline/prepare.py``)."""
    from .pipeline.prepare import prepare as run
    run(cfg, paths)


def build_relevance(cfg: BuildConfig, paths: DatasetPaths, limit: Optional[int] = None) -> None:
    """Stage data_relevance of the dataset of *paths*
    (``pipeline/relevance.py``)."""
    from .pipeline.relevance import build_relevance as run
    run(cfg, paths, limit)
