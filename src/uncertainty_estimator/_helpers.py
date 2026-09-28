"""Small helpers shared by the uncertainty estimator modules."""

import json
import re
from typing import Any, Dict, Optional


def doc_id(doc: Dict[str, Any]) -> str:
    """The document id, from ``doc_id`` or ``id``; "" when neither is set."""
    return doc.get("doc_id") or doc.get("id") or ""


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from LLM output, with or without a code fence."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None
