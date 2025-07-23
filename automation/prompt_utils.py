from __future__ import annotations

from typing import Iterable, Tuple
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
