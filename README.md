# DRA Uncertainty

Uncertainty quantification for **Deep Research Agents (DRA)**: agentic retrieval-augmented models that iterate
*think → search → observe → answer/report* over a document index.

## Task

A deep research agent answers a request by searching many times before it writes its answer or report. It gives no
sign of how far its search has come or how much its final answer can be trusted. The goal of this project is an
uncertainty score for the agent, both during its search and for its final answer.

The project has two parts:

| Part | What it does | README |
|---|---|---|
| Inference | runs an agent on a dataset with the belief model (`off` / `monitor` / `inform`), evaluates retrieval, answers or reports and the trajectory, and saves the belief state and signals of every step | [experiments/inference/README.md](experiments/inference/README.md) |
| Training | trains the belief model with the agent frozen: training pools, teacher runs, SFT, then RL | [experiments/training/README.md](experiments/training/README.md) |

## Supported agents, datasets and retrievers

**Agents** (the LLM is picked automatically per agent, `utils/config.py`):

| Family | Agents |
|---|---|
| Instruction-tuned (API) | `react`, `selfask`, `searcho1` (claude-sonnet-4-6); `belief_aware` (qwen3.6-27b, reads the belief tag in `inform` mode); `webweaver` (qwen3.6-35b-a3b) |
| RL-trained (vLLM) | `searchr1`, `research`, `stepsearch`, `drtulu`, `glm`, `oss_20b`, `oss_120b`, `tongyi`, `cpm_explore` |
| Report (vLLM) | `cpm_report` |

**Datasets:** `trqa` (Wikipedia / e-commerce), `neuclir` (news + technical, 2022–2024), `browsecomp_plus`,
`ragtime` (multilingual news, 2025).

**Retrievers:** `bm25`, `spladepp`, `spladev3` (sparse); `bge`, `e5`, `dpr`, `contriever`,
`qwen3_emb_{0.6b,4b,8b}`, `agentir_4b` (dense); `rerank_l6`, `rerank_l12` (rerankers).

## Repository layout

```
src/
├── deep_research_agents/     agents and their prompts
├── searcher_component/       retrievers, rerankers, fusion
├── reasoner_component/       LLM generators (API, vLLM)
├── orchestration/            wiring of agent, searcher and belief model; multi-GPU workers
├── belief_model_component/   criteria, belief state, signals, judges
├── evaluation/               answer, retrieval, trajectory and generation evaluators
├── indexing_corpus_dataset/  dataset download and index building
└── belief_model_training/    training phases of the belief model
experiments/
├── inference/                dra_inference.py
├── training/                 build_training_pool.py, belief_model_training.py
└── configs/                  YAML defaults of every script
analysis/                     criteria reachability, signal correlation
scripts/                      Slurm wrappers (run_*.sh)
data/                         criteria banks, training pools
```

## Installation

```bash
pip install -e .
```

This registers all packages so imports resolve from any working directory. Set `DRA_DATA_ROOT` (corpus + indices),
`DRA_OUTPUT_ROOT` (run outputs) and `DRA_CRITERIA_ROOT` (criteria banks, default the repo's `data/`) to control
where data is read and written. On Slurm, `scripts/_activate.sh` loads the modules, activates the venv and exports
these variables.

## Data and indices

Inference and training both read the datasets and indices built here.

### Download datasets

```bash
# trqa  (subset: wiki1 | wiki2 | ecommerce)
python src/indexing_corpus_dataset/download_datasets.py trqa --subset wiki1

# neuclir  (subset: news | technical; 2024 news also writes report nuggets)
python src/indexing_corpus_dataset/download_datasets.py neuclir --year 2024 --subset news

# ragtime  (2025; queries, qrels, nuggets, English corpus)
python src/indexing_corpus_dataset/download_datasets.py ragtime

# browsecomp_plus  (corpus only)
python src/indexing_corpus_dataset/download_datasets.py browsecomp_plus --skip-queries-qrels
```

Canonical layout written under `$DRA_DATA_ROOT/<dataset>/`:

```
queries/queries_{split}.jsonl     {"id","text","answer"?}
qrels/qrels_{split}.txt           TREC: qid 0 docid rel
nuggets/nuggets_{split}.jsonl     {"id","nuggets":[{"id","question","answers","importance","support_docs"}]}
corpus/{name}.jsonl               {"id","contents"}
```

### Build an index

```bash
python -m indexing_corpus_dataset.index_builder \
    --retriever qwen3_emb_4b --dataset browsecomp_plus \
    --use_fp16 --max_length 4096 --batch_size 16 --faiss_type Flat --save_embedding
```

On Slurm: `sbatch scripts/run_index_builder.sh` (sets per-retriever arguments automatically). Build defaults live in
`src/indexing_corpus_dataset/configs/index_build.yaml`.

To test a build, a small smoke index over gold and distractor passages checks entity recall:

```bash
python src/indexing_corpus_dataset/index_builder_test.py \
    --retriever bge --dataset neuclir --query-limit 20
```

## Next steps

- Run an agent with and without the belief model in inference time: [Link](experiments/inference/README.md)
- Train the belief model: [Link](experiments/training/README.md)
