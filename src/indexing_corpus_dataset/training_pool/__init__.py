"""Training pool: deep-research questions with several gold pages each,
merged from InfoSeek-Hard, WebShaper and FRAMES, with their gold pages
fetched from the source.  Shared by the trainings that need such questions
(first the belief model, phase 0).

Steps (``experiments/training/build_training_pool.py --step N``) under
``{REPO_DATA_ROOT}/training_pool/`` (the repo's data/):

    1  questions      merge the sources, drop duplicates and the fixed
                      question types: questions.jsonl
    2  gold pages     pages/index.jsonl + pages/text.jsonl, then Filter A
                      (Wikipedia-only questions: every gold page; with web
                      gold pages: >= min_page_share of all of them); the
                      kept questions in the canonical layout:
                      queries/queries_train.jsonl, qrels/qrels_train.txt
                      (binary, one line per fetched gold page),
                      corpus/corpus.jsonl (those pages); incremental
                      (--limit N = the next N unprocessed questions,
                      --source to pick sources)
    3  stats          the dataset as written, per source, and consistency
                      checks between queries, qrels and corpus

Every step writes its section of ``stats.json`` (step 2 with its settings);
steps 1 and 2 write their dropped questions with their reasons to
``dropped.jsonl``.

Modules:
    config         settings (shared, pages) and the build paths
    sources        loaders of the three sources -> one record format
    http_client    sessions, User-Agent, retries
    wikipedia_api  title resolution (redirects) and page HTML
    web_pages      live fetch + Trafilatura / pypdf, Wayback Machine fallback
    filters        type filter (step 1), Filter A (step 2)
    stats          the step statistics
    steps          run_step1 / run_step2 (resume, progress) / run_step3
"""
