"""Shared helpers of every package: a leaf, it imports none of the other
packages under ``src/`` (only the standard library and light third-party ones).

Modules:
    paths        the repo, data, output and repo-data roots (``DRA_*`` env vars, ``.env``)
    models       policy and belief model maps, sampling settings, OpenRouter routing
    documents    doc dicts: id, title, text, the token cut of a passage
    citations    an agent result's ranked cited docs and its References section
    json_parse   a JSON object out of LLM output
    io           JSONL reading and writing, atomic writes, a file lock
    yaml_config  sectioned YAML configs, flat configs with CLI overrides
    progress     progress bars of the run stages
    token_meter  token accounting of the LLM clients
    run_files/   every file of a run directory (written and read)
"""
