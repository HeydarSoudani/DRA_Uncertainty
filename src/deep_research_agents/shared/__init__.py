"""What every agent uses.

    config       InferenceConfig: the inference settings an agent passes to its LLM calls
    formatting   text the agents show and print: verbose loop lines, documents for the
                 prompt, numbered citations, the evidence summary of a reasoning path
    parsing      the agents' LLM output: XML tags, function-call actions, tool calls

Nothing here knows an agent: the agents import from here, never the reverse.
"""
