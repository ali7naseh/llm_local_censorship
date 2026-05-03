"""
Template for calling a locally hosted Hugging Face causal language model.

This file illustrates the common wrapper interface used for GPU-based
generation. Model identifiers and cache paths should be configured locally.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.environ.get("HF_MODEL_ID", "organization/model-name")
CACHE_DIR = os.environ.get("HF_CACHE_DIR", None)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    device_map="auto",
    torch_dtype="auto",
)

model.eval()


def split_thinking_and_answer(text: str) -> Tuple[str, str]:
    """
    Extract reasoning traces enclosed in <think>...</think>, when present.

    Returns:
        thinking: extracted reasoning trace, or ""
        final_answer: remaining final response
    """

    text = text or ""

    match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
    if not match:
        return "", text.strip()

    thinking = match.group(1).strip()
    final_answer = match.group(2).strip()
    return thinking, final_answer


def call_hf_gpu_model(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: Optional[float] = None,
    do_sample: bool = True,
) -> Dict[str, Any]:
    """
    Call a locally hosted Hugging Face chat model and normalize the output.

    Returns:
        {
            "model": str,
            "text": str,       # final response
            "thinking": str,   # reasoning trace if available, otherwise ""
            "raw_text": str,   # raw generated text
            "usage": dict,     # approximate token usage
        }
    """

    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    gen_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
    }

    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]

    raw_generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=False,
    )

    thinking, final_text = split_thinking_and_answer(raw_generated_text)

    return {
        "model": MODEL_ID,
        "text": final_text,
        "thinking": thinking,
        "raw_text": raw_generated_text,
        "usage": {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(len(generated_tokens)),
            "total_tokens": int(outputs.shape[1]),
        },
    }


if __name__ == "__main__":
    result = call_hf_gpu_model(
        user_prompt="What is the capital of France?",
        max_new_tokens=512,
    )

    print("Thinking:\n", result["thinking"])
    print("\nAnswer:\n", result["text"])