"""Training pools, one folder each under ``{TRAINING_POOL_ROOT}/{pool}``
(data/training_pool/ of the repo); each pool is a dataset of dra_inference.

Every pool has the split ``all`` (every query) and its split into ``sft``,
``rl`` and ``val`` by a hash of the query id (``splits.py``), for belief
model training.

trqa_validation (``build_training_pool.py --pool trqa_validation``): the
TRQA wiki1 + wiki2 validation files merged (``trqa_validation.py``), then
split.

deep_qa: deep-research questions with several gold pages each, merged from
InfoSeek-Hard, WebShaper and FRAMES, with their gold pages fetched from the
source.  Steps (``experiments/training/build_training_pool.py --step N``):

    1  questions      merge the sources, drop duplicates and the fixed
                      question types: questions.jsonl
    2  gold pages     pages/index.jsonl + pages/text.jsonl, then Filter A
                      (Wikipedia-only questions: every gold page; with web
                      gold pages: >= min_page_share of all of them); the
                      kept questions in the canonical layout:
                      queries/queries_all.jsonl, qrels/qrels_all.txt
                      (binary, one line per fetched gold page),
                      corpus/corpus.jsonl (those pages and their hard
                      negatives, corpus.py); then the split:
                      queries/queries_{sft,rl,val}.jsonl,
                      qrels/qrels_{sft,rl,val}.txt (``splits.py``; a query
                      keeps its split as the pool grows); incremental
                      (--limit N = the next N unprocessed questions,
                      --source to pick sources)
    3  negatives      hard negatives of the Wikipedia gold pages from the
                      Wikipedia API (morelike, category siblings; at most
                      negatives.max_per_gold each): negatives/index.jsonl
                      (the memory: one row per processed gold page), their
                      text in pages/text.jsonl, then corpus/corpus.jsonl
                      (gold + hard negatives), corpus/ids_gold.txt,
                      corpus/ids_hard_negative.txt; incremental (--limit N =
                      the next N unprocessed gold pages, --source to pick
                      sources)
    4  stats          the dataset as written, per source, the hard
                      negatives, and consistency checks between queries,
                      qrels, corpus and id files

Every step writes its section of ``stats.json`` (steps 2 and 3 with their settings,
the split its shares and counts in ``split``); steps 1 and 2 write their
dropped questions with their reasons to ``dropped.jsonl``.

Modules:
    config         settings (shared, pages, negatives, split) and the build paths
    sources        loaders of the three sources -> one record format
    http_client    sessions, User-Agent, retries
    wikipedia_api  title resolution (redirects), page HTML, candidate negatives
    web_pages      live fetch + Trafilatura / pypdf, Wayback Machine fallback
    filters        type filter (step 1), Filter A (step 2)
    stats          the step statistics
    steps          run_step1 / run_step2 (resume, progress) / run_step4
    negatives      run_step3 (hard negatives, memory, resume)
    corpus         the corpus and id files, written by steps 2 and 3
    trqa_validation  the merged TRQA validation pool
    splits         the sft / rl / val split of a pool
"""
