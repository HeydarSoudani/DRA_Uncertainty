"""Training datasets, one folder each under ``{TRAINING_DATASETS_ROOT}/{dataset}``
(data/training_datasets/ of the repo); each is a dataset of dra_inference.

Every training dataset has one row per query in ``queries/queries_all.jsonl``
(the question, answer, source, split and gold annotation, ``gold/annotations.py``)
and the qrels in ``qrels/qrels_all.txt``.  The splits ``sft``, ``rl`` and
``val`` (``splits.py``) are views of those files, read by
``dataset_loaders``.  One build call per dataset
(``experiments/training/train_belief_model.py --stage data --dataset D --limit N``),
incremental: the next N questions not processed yet.

infoseek_hard (``infoseek_hard.py``): InfoSeek-Hard questions over the corpus
of the original release, wiki-25-512 (Wikipedia 2025 in chunks of 512 tokens,
downloaded once into DATA_ROOT/infoseek/corpus); the qrels are the first
chunk of each gold page (matched by title), a hop's documents the first
chunks of its page and of the pages linking to it; a question
with a gold page missing from the corpus is dropped.  Split sft / rl / val.

trqa_val (``trqa_val.py``): the TRQA wiki1 + wiki2 validation files merged
with their gold annotation, its belief texts and the split sft / rl / val;
it searches TRQA's corpus.

webshaper_frames (``webshaper_frames/``, one module per stage: questions,
pages, negatives, report): WebShaper and FRAMES questions with their gold
pages fetched from the source and hard negatives of the Wikipedia gold
pages, validation only.  Kept in the code but not registered (no
``layout.DATASET_SPECS`` entry, not a stage data choice): it does not build.

Every build writes its sections of ``stats.json`` (its stages with their
settings, ``annotations``, the split's shares and counts in ``split``) and
its dropped questions with their stage and reasons to ``dropped.jsonl``.

Modules:
    config            settings (split, gold_beliefs; shared, pages, negatives
                      of webshaper_frames) and the build paths
    infoseek_hard     the infoseek_hard build (corpus download, title -> chunks)
    trqa_val          the trqa_val build (merged TRQA validation files)
    webshaper_frames/ the unregistered webshaper_frames build, one module per
                      stage with its corpus, duplicates, filters and web
                      fetchers (fetch/)
    sources           loaders of the sources -> one record format
    splits            the sft / rl / val split of a dataset (the rows' ``split``)
    stats             the build statistics
    gold/             the gold annotation: row format, level and skeletons
                      (annotations), the belief texts by an LLM
                      (belief_texts, prompts/), the TRQA gold beliefs
                      for the belief model (trqa)
"""
