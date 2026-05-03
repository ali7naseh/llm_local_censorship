"""
Template for calling an API-hosted chat model.

This file illustrates the common wrapper interface used in the experiments.
Provider-specific endpoints, deployment names, and credentials are omitted.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


# Configure these through environment variables or a local config file.
# Do not hard-code credentials or provider-specific private endpoints.
ENDPOINT = os.environ.get("LLM_API_ENDPOINT", "https://example-provider-endpoint/v1/")
DEPLOYMENT = os.environ.get("LLM_API_DEPLOYMENT", "example-model-deployment")
API_KEY = os.environ.get("LLM_API_KEY", "")


client = OpenAI(
    base_url=ENDPOINT,
    api_key=API_KEY,
)


def call_api_model(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call an API-hosted chat model and normalize the output.

    Returns:
        {
            "model": str,
            "text": str,       # final response
            "thinking": str,   # reasoning trace if available, otherwise ""
            "usage": dict,     # token usage if available
        }
    """

    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    kwargs: Dict[str, Any] = {
        "model": DEPLOYMENT,
        "messages": messages,
    }

    if max_tokens is not None:
        # Some providers use max_tokens; others use max_completion_tokens.
        kwargs["max_completion_tokens"] = max_tokens

    if temperature is not None:
        kwargs["temperature"] = temperature

    if top_p is not None:
        kwargs["top_p"] = top_p

    completion = client.chat.completions.create(**kwargs)
    msg = completion.choices[0].message

    final_text = msg.content or ""

    # Some reasoning APIs expose a separate reasoning_content field.
    reasoning_text = getattr(msg, "reasoning_content", None) or ""

    usage: Dict[str, Any] = {}
    if getattr(completion, "usage", None):
        usage = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
            "completion_tokens": getattr(completion.usage, "completion_tokens", None),
            "total_tokens": getattr(completion.usage, "total_tokens", None),
        }

        completion_details = getattr(completion.usage, "completion_tokens_details", None)
        if completion_details is not None:
            usage["reasoning_tokens"] = getattr(completion_details, "reasoning_tokens", None)

    return {
        "model": getattr(completion, "model", DEPLOYMENT),
        "text": final_text,
        "thinking": reasoning_text,
        "usage": usage,
    }


if __name__ == "__main__":
    result = call_api_model("What is the capital of France?")
    print("Thinking:\n", result["thinking"])
    print("\nAnswer:\n", result["text"])