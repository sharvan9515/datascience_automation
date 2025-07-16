from __future__ import annotations

from typing import Iterable, Tuple
import os


def query_llm(prompt: str, few_shot: Iterable[Tuple[str, str]] | None = None) -> str:
    """Call the OpenAI chat completion API with optional few-shot messages."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    try:
        import openai
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("openai package is required") from exc

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

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LLM call failed: {exc}") from exc
