"""What the searcher's modules use.

    config      the local retrievers' settings (RetrieverConfig) and the rerankers'
    normalize   a retriever response as one document schema (doc_id, relevant_text, title)
    pooling     a dense encoder's output as one embedding (retrievers and index builder)

``config`` and ``normalize`` are light (no torch, FAISS or sentence-transformers):
the belief and eval stages import ``normalize`` from here on CPU.  ``pooling``
needs torch.
"""
