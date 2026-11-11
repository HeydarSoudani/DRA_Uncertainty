QUERY_TEMPLATE = """Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

# Report tasks (layout.DATASET_SPECS task == "report"): the response is the
# report itself; its docid citations become numbered [N] markers
# (utils.text_utils.number_docid_citations).
REPORT_QUERY_TEMPLATE = """Request: {Question}

Your response should be the report that answers the request. Cite your evidence documents inline by enclosing their full docids, exactly as given in the search results, in square brackets [] at the end of sentences. For example, [<docid>]. Do not add a reference list."""
