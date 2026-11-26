"""Parse a JSON object out of LLM output (with or without a code fence)."""

import json
import re
from typing import Any, Dict, List, Optional


def _escape_inner_quotes(text: str) -> str:
    """Escape the double quotes inside JSON strings that the model left raw
    (e.g. a quotation copied verbatim into a span).  A quote inside a string
    ends it only when the next non-space character is , : } ] or the end."""
    out: List[str] = []
    in_string = escaped = False
    for i, ch in enumerate(text):
        if not in_string:
            in_string = ch == '"'
            out.append(ch)
            continue
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            rest = text[i + 1:].lstrip()
            if not rest or rest[0] in ",:}]":
                in_string = False
            else:
                out.append("\\")
        out.append(ch)
    return "".join(out)


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from LLM output, with or without a code fence;
    unescaped quotes inside strings are repaired as a last resort."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    candidates = [text]
    if 0 <= start < end:
        body = text[start:end + 1]
        candidates += [body, _escape_inner_quotes(body)]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else None
    return None
