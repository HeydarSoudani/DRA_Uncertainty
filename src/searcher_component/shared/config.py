"""Configuration of the searcher: the local retrievers' settings and the rerankers'."""


class RetrieverConfig:
    """Lightweight config object for local retrievers (BM25, Dense, Rerank, SPLADE)."""

    def __init__(self, retriever_name, index_dir, corpus_path, topk):
        self.retriever_name = retriever_name
        self.index_dir = index_dir
        self.corpus_path = corpus_path
        self.retrieval_topk = topk
        self.bm25_k1 = 0.9
        self.bm25_b = 0.4
        self.faiss_gpu = False
        self.retrieval_query_max_length = 512
        self.retrieval_use_fp16 = True
        # SPLADE-specific
        self.splade_max_length = 256
        self.device = None  # auto-detected by SPLADERetriever


def get_reranker_configs(rerank_top_k: int = 100) -> dict:
    """Return the canonical reranker configuration dict."""
    return {
        "batched_reranker_config": {
            "reranker_model": "claude-sonnet-4-5",
            "max_chars_per_document": 4096,
            # Optional score-threshold selector (dynamic-length output). Disabled
            # by default; set enable_selector=True to keep only high-scoring docs.
            "enable_selector": False,
            "selector_score_threshold": 3.0,
            "selector_min_keep": 0,
        },
        "rankllama_reranker_config": {
            "model_name": "castorini/rankllama-v1-7b-lora-passage",
            "top_k": rerank_top_k,
            "batch_size": 1,
            "rerank_max_len": 256,
        },
        "rank1_reranker_config": {
            "model_name": "jhu-clsp/rank1-7b",
            "backend": "api",
            "api_url": "http://localhost:8000/v1",
            "api_model_name": "jhu-clsp/rank1-7b",
            "top_k": rerank_top_k,
            "batch_size": 100,
            "context_size": 128,
            "max_output_tokens": 200,
        },
        "qwen3_reranker_config": {
            "model_name": None,
            "size": "4B",
            "api_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
            "batch_size": 32,
            "top_k": rerank_top_k,
            "enable_thinking": False,
        },
        "listwise_reranker_config": {
            "model_name": "castorini/rank_zephyr_7b_v1_full",
            "api_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
            "template": "rankzephyr",
            "top_k": rerank_top_k,
            "window_size": 20,
            "stride": 10,
            "use_sliding_window": True,
            "max_passage_words": 300,
            "max_tokens": 200,
            "temperature": 0.0,
            "enable_thinking": False,
        },
        "rank_r1_reranker_config": {
            "api_url": "http://localhost:8001/v1",
            "api_model_name": "rank-r1",
            "api_key": "EMPTY",
            "num_child": 19,
            "k": 10,
            "max_tokens": 2048,
            "top_k": rerank_top_k,
            "context_size": 450,
        },
        "monot5_reranker_config": {
            "model_name": "castorini/monot5-base-msmarco",
            "use_mt5": False,
            "batch_size": 32,
            "top_k": rerank_top_k,
        },
    }
