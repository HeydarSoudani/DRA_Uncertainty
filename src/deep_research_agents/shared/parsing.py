"""Parse the agents' LLM output: XML tags, function-call actions and tool calls."""

import json
import re
from typing import Any, Dict, List, Optional

try:
    import json5 as _json5
    _HAS_JSON5 = True
except ImportError:
    _HAS_JSON5 = False


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract the content of the first ``<tag>…</tag>`` block in *text*."""
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def extract_all_tag_content(text: str, tag: str) -> List[str]:
    """Extract content from *all* ``<tag>…</tag>`` blocks in *text*."""
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    return [m.strip() for m in pattern.findall(text)]


def get_think(text: str) -> Optional[str]:
    return extract_tag_content(text, "think")


def get_query(text: str) -> Optional[str]:
    return extract_tag_content(text, "search")


def get_answer(text: str) -> Optional[str]:
    return extract_tag_content(text, "answer")


def get_action(text: str) -> Optional[str]:
    return extract_tag_content(text, "action")


def parse_action_call(text: str) -> Optional[tuple]:
    """Parse ``search(query="...")`` or ``finish(answer="...")`` function-call syntax."""
    if not text:
        return None
    text = text.strip()
    for func, param in [("search", "query"), ("finish", "answer")]:
        for quote in ['"', "'"]:
            m = re.match(rf'{func}\s*\(\s*{param}\s*=\s*{quote}(.*){quote}\s*\)', text, re.DOTALL)
            if m:
                return (func, m.group(1).strip())
    for func in ["search", "finish"]:
        for quote in ['"', "'"]:
            m = re.match(rf'{func}\s*\(\s*{quote}(.*){quote}\s*\)', text, re.DOTALL)
            if m:
                return (func, m.group(1).strip())
    return None


def parse_tool_call_xml(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from ``<tool_call>…</tool_call>``."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip()
    if _HAS_JSON5:
        try:
            return _json5.loads(raw)
        except Exception:
            pass
    try:
        return json.loads(raw)
    except Exception:
        return None


def parse_tool_calls_xml_list(text: str) -> List[Dict[str, Any]]:
    """Extract and parse *all* ``<tool_call>…</tool_call>`` blocks."""
    calls = []
    for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        if _HAS_JSON5:
            try:
                calls.append(_json5.loads(raw))
                continue
            except Exception:
                pass
        try:
            calls.append(json.loads(raw))
        except Exception:
            continue
    return calls


def extract_think_and_clean(text: str) -> tuple:
    """Extract ``<think>`` or ``<thought>`` content and return (reasoning, cleaned_text)."""
    for tag in ("think", "thought"):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        m = pattern.search(text)
        if m:
            return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    return "", text
