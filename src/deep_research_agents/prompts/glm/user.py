QUERY_TEMPLATE = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# Report tasks (layout.DATASET_SPECS task == "report"): the response is the
# report itself (see prompts.oss.user.REPORT_QUERY_TEMPLATE).
REPORT_QUERY_TEMPLATE = """
You are a deep research agent. You need to write a report for the given request by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Request: {Question}

Your response should be the report that answers the request. Cite your evidence documents inline by enclosing their full docids, exactly as given in the search results, in square brackets [] at the end of sentences. For example, [<docid>]. Do not add a reference list.
""".strip()
