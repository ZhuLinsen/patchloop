"""JSON 解析辅助。"""
from __future__ import annotations

import json
from typing import Any


def extract_json_object(raw: str, *, context: str) -> dict[str, Any]:
    text = _strip_markdown_fence(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _decode_first_json_object(str(raw or ""))
        if data is None:
            raise ValueError(f"无法从 {context} 中解析 JSON: {str(raw or '')[:200]}")
    if not isinstance(data, dict):
        raise ValueError(f"{context} 不是 JSON object")
    return dict(data)


def _strip_markdown_fence(raw: str) -> str:
    text = str(raw or "").strip()
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def _decode_first_json_object(raw: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    text = str(raw or "")
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return dict(candidate)
    return None
