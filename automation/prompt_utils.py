from __future__ import annotations

from typing import Iterable, Tuple, Dict, Any
import os
import re
import time
import json


def query_llm(
    prompt: str,
    few_shot: Iterable[Tuple[str, str]] | None = None,
    expect_json: bool = False,
    *,
    timeout: float | None = 30.0,
    max_retries: int = 3,
    json_schema: dict | None = None,
) -> str:
    """Call the OpenAI chat completion API with optional few-shot messages.

    Parameters
    ----------
    prompt:
        The user prompt to send to the LLM.
    few_shot:
        Optional list of (user, assistant) message tuples to include as
        examples.
    expect_json:
        If ``True``, request a JSON-formatted response from the LLM.
    timeout:
        Request timeout in seconds passed to the OpenAI client.
    max_retries:
        Number of attempts before giving up.
    json_schema:
        Optional JSON schema to validate the response against. Requires the
        ``jsonschema`` package if supplied.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    try:
        import openai
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("openai package is required") from exc

    if json_schema is not None:
        try:
            import jsonschema  # noqa: WPS433 -- optional dependency
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "jsonschema package is required for validation"
            ) from exc
    else:
        jsonschema = None  # type: ignore

    client = openai.OpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data science automation assistant. "
                "Follow instructions carefully and respond only in the requested format."
            ),
        }
    ]

    if few_shot:
        for user_msg, assistant_msg in few_shot:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.0,
        "timeout": timeout,
    }
    if expect_json:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"```(?:json)?\n(.*?)```", raw, re.S)
            if match:
                raw = match.group(1).strip()
            if json_schema is not None and jsonschema is not None:
                try:
                    parsed = json.loads(raw)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"LLM response JSON parse failed: {exc}"
                    ) from exc
                try:
                    jsonschema.validate(instance=parsed, schema=json_schema)
                except jsonschema.ValidationError as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"LLM response failed validation: {exc.message}"
                    ) from exc
            return raw
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"LLM call failed after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(2 ** attempt)


def create_context_aware_prompt(
    profile: Dict[str, Any] | None,
    task_type: str,
    stage: str,
    algorithms: list[str] | None = None,
) -> str:
    """Return a dataset-aware context string for LLM prompts."""

    lines: list[str] = [f"You are assisting with {stage} for a {task_type} task."]
    if not profile:
        return "\n".join(lines)

    stats = profile.get("statistical_summary", {})
    n_cols = len(stats)
    n_rows = 0
    for metrics in stats.values():
        count = metrics.get("count")
        if isinstance(count, (int, float)):
            n_rows = max(n_rows, int(count))
    if n_rows and n_cols:
        lines.append(f"Dataset shape: {n_rows} rows x {n_cols} columns.")

    missing = profile.get("missing_patterns", {}).get("column_summary", {})
    miss_counts = {
        k: v.get("missing_count", 0)
        for k, v in missing.items()
        if v.get("missing_count", 0) > 0
    }
    outliers = profile.get("outlier_detection", {})
    out_total = sum(outliers.values()) if outliers else 0
    dq_parts: list[str] = []
    if miss_counts:
        dq_parts.append(f"missing values {dict(list(miss_counts.items())[:3])}")
    if out_total:
        dq_parts.append(f"outliers {out_total}")
    if dq_parts:
        lines.append("Data quality issues: " + ", ".join(dq_parts) + ".")

    cm = profile.get("complexity_metrics", {})
    cm_parts: list[str] = []
    if "feature_target_ratio" in cm:
        cm_parts.append(f"feature/row ratio {cm['feature_target_ratio']:.3f}")
    if cm.get("noise_level") is not None:
        cm_parts.append(f"noise level {cm['noise_level']:.3f}")
    if cm.get("class_imbalance"):
        cm_parts.append(f"class imbalance {cm['class_imbalance']}")
    if cm_parts:
        lines.append("Complexity metrics: " + ", ".join(cm_parts) + ".")

    domain = profile.get("domain_insights", {})
    domain_parts: list[str] = []
    sem = domain.get("column_semantics")
    if sem:
        domain_parts.append(f"semantics {dict(list(sem.items())[:3])}")
    temporal = domain.get("temporal_patterns")
    if temporal:
        domain_parts.append(f"temporal {dict(list(temporal.items())[:1])}")
    cardinality = domain.get("categorical_cardinality")
    if cardinality:
        domain_parts.append(f"categorical cardinality {dict(list(cardinality.items())[:3])}")
    if domain_parts:
        lines.append("Domain characteristics: " + ", ".join(domain_parts) + ".")

    if algorithms:
        lines.append("Recommended algorithms: " + ", ".join(algorithms) + ".")

    return "\n".join(lines)
