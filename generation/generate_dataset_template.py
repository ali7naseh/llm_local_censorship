"""
Generic dataset-generation template.

This script applies a model wrapper to each prompt in a dataset and stores
the generated response and optional reasoning trace. The model wrapper may
call an API-hosted model or a locally hosted GPU model, as long as it exposes
a function that returns a normalized dictionary with fields such as:

    {
        "text": str,
        "thinking": str,
        "usage": dict,
    }

Provider-specific wrappers are omitted from this artifact.
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from typing import Any, Callable, Dict, Tuple

import pandas as pd
from tqdm import tqdm


def load_call_function(module_name: str, function_name: str) -> Callable[[str], Dict[str, Any]]:
    """
    Dynamically load a model-call function.

    Example:
        module_name = "api_model_wrapper_template"
        function_name = "call_api_model"
    """
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def normalize_output(out: Any) -> Tuple[str, str, Dict[str, Any]]:
    """
    Normalize model-wrapper output into:
        text, thinking, usage

    Wrappers are expected to return a dictionary, but plain-text outputs are
    also supported as a fallback.
    """
    if isinstance(out, dict):
        text = (out.get("text", "") or "").strip()
        thinking = (out.get("thinking", "") or "").strip()
        usage = out.get("usage", {}) or {}
    else:
        text = str(out).strip()
        thinking = ""
        usage = {}

    return text, thinking, usage


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--prompt_col", type=str, default="Prompt")
    parser.add_argument("--response_col", type=str, default="Generated_Response")
    parser.add_argument("--thinking_col", type=str, default="Generated_Thinking")
    parser.add_argument("--error_col", type=str, default="Generation_Error")

    parser.add_argument(
        "--wrapper_module",
        type=str,
        required=True,
        help="Python module containing the model-call function.",
    )
    parser.add_argument(
        "--wrapper_function",
        type=str,
        required=True,
        help="Function name for calling the model.",
    )

    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--sleep_base", type=float, default=1.5)
    parser.add_argument("--sleep_max", type=float, default=30.0)

    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)

    args = parser.parse_args()

    call_model = load_call_function(args.wrapper_module, args.wrapper_function)

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(args.output_file):
        df = pd.read_pickle(args.output_file)
    else:
        df = pd.read_pickle(args.input_file)

    for col in [args.response_col, args.thinking_col, args.error_col]:
        if col not in df.columns:
            df[col] = None

    n = len(df)
    s = 0 if args.start_idx is None else max(0, args.start_idx)
    e = n if args.end_idx is None else min(n, args.end_idx)

    if s >= e:
        raise ValueError(f"Empty subset: start={s}, end={e}, len(df)={n}")

    sub = df.iloc[s:e]

    missing_mask = sub[args.response_col].isna()
    if not missing_mask.any():
        print(f"Subset already complete: rows [{s}, {e})")
        return

    processed_since_save = 0

    for idx, row in tqdm(
        sub.iterrows(),
        total=len(sub),
        desc=f"Generating rows [{s}, {e})",
    ):
        if pd.notna(df.at[idx, args.response_col]):
            continue

        prompt = row[args.prompt_col]

        final_text = ""
        final_thinking = ""
        final_error = None

        for attempt in range(args.max_attempts):
            try:
                out = call_model(prompt)
                final_text, final_thinking, _usage = normalize_output(out)
                final_error = None
                break

            except Exception as ex:
                final_text = ""
                final_thinking = ""
                final_error = f"{type(ex).__name__}: {ex}"

                sleep_s = min(args.sleep_max, args.sleep_base * (2 ** attempt))
                time.sleep(sleep_s)

        df.at[idx, args.response_col] = final_text
        df.at[idx, args.thinking_col] = final_thinking
        df.at[idx, args.error_col] = final_error

        processed_since_save += 1
        if processed_since_save >= args.save_every:
            df.to_pickle(args.output_file)
            processed_since_save = 0

    df.to_pickle(args.output_file)
    print(f"Done. Saved to: {args.output_file} (subset rows [{s}, {e}))")


if __name__ == "__main__":
    main()