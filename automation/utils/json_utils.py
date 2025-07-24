import json
import ast
import re
from typing import Any


def safe_json_parse(text: str) -> Any:
    """Return JSON parsed from ``text`` with best-effort fixes."""
    try:
        return json.loads(text)
    except Exception:
        # Replace single quotes with double quotes and remove trailing commas
        cleaned = text.replace("'", '"')
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse JSON: {exc}. Raw: {text}")

