"""Training datasets, one folder each under ``{TRAINING_DATASETS_ROOT}/{dataset}``
(data/training_datasets/ of the repo); each is a dataset of dra_inference.

Every training dataset has one row per query in ``queries/queries_all.jsonl``
(the question, answer, source, split and gold annotation, ``gold/annotations.py``)
and the qrels in ``qrels/qrels_all.txt``.  The splits ``sft``, ``rl`` and
``val`` (``splits.py``) are views of those files, read by
``dataset_loaders``.  One build call per dataset
(``experiments/training/build_training_dataset.py --dataset D --limit N``),
incremental: the next N questions not processed yet.

infoseek (``infoseek.py``): InfoSeek-Hard questions over the corpus of the
original release, wiki-25-512 (Wikipedia 2025 in chunks of 512 tokens,
downloaded once into DATA_ROOT/infoseek/corpus); the qrels are the first
chunk of each gold page (matched by title), a hop's documents the first
chunks of its page and of the pages linking to it; a question
with a gold page missing from the corpus is dropped.  Split sft / rl / val.

webshaper_frames (``webshaper_frames/``, one module per stage): WebShaper and FRAMES questions
with their gold pages fetched from the source and hard negatives of the
Wikipedia gold pages, in stages:

    questions  merge the sources, drop duplicates and the fixed question
               types: questions.jsonl (each record with the skeleton of its
               gold annotation, gold/annotations.py)
    pages      pages/index.jsonl + pages/text.jsonl, then Filter A
               (Wikipedia-only questions: every gold page; with web gold
               pages: >= min_page_share of all of them); the kept questions
               in the canonical layout: queries/queries_all.jsonl (one row
               per question with its gold annotation, the criterion texts by
               an LLM, gold/criterion_texts.py, kept in criterion_texts.jsonl, and
               its split: every query in val), qrels/qrels_all.txt (binary,
               one line per fetched gold page), corpus/corpus.jsonl (those
               pages and their hard negatives, webshaper_frames/corpus.py)
    negatives  hard negatives of the new Wikipedia gold pages from the
               Wikipedia API (morelike, category siblings; at most
               negatives.max_per_gold each): negatives/index.jsonl (the
               memory: one row per processed gold page), their text in
               pages/text.jsonl, then corpus/corpus.jsonl (gold + hard
               negatives), corpus/ids_gold.txt, corpus/ids_hard_negative.txt
    report     the dataset as written, per source, the hard negatives, the
               gold annotation, consistency checks between queries, qrels,
               corpus and id files, and the duplicate documents of the corpus

trqa_validation (``trqa_validation.py``): the TRQA wiki1 + wiki2 validation
files merged with their gold annotation, its criterion texts and the split
sft / rl / val; it searches TRQA's corpus.

Every build writes its sections of ``stats.json`` (its stages with their
settings, ``annotations``, the split's shares and counts in ``split``) and
its dropped questions with their stage and reasons to ``dropped.jsonl``.

Modules:
    config            settings (shared, pages, negatives, split, gold_criteria)
                      and the build paths
    infoseek          the infoseek build (corpus download, title -> chunks)
    trqa_validation   the trqa_validation build (merged TRQA validation files)
    webshaper_frames/ the webshaper_frames build, one module per stage
                      (questions, pages, negatives, report) with its corpus,
                      duplicates, filters and web fetchers (fetch/)
    sources           loaders of the sources -> one record format
    splits            the sft / rl / val split of a dataset (the rows' ``split``)
    stats             the build statistics
    gold/             the gold annotation: row format, level and skeletons
                      (annotations), the criterion texts by an LLM
                      (criterion_texts, prompts/), the TRQA gold criteria
                      for the belief model (trqa)
"""
