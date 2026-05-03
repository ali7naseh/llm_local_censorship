# llm_local_censorship

This repository contains anonymized artifact materials associated with the submission. The artifact provides a lightweight implementation of the censorship measurement framework used in the paper, including prompt templates, generation templates, judge-score calibration utilities, and analysis scripts for global, model-level local, and group-level censorship.

The repository is intended to document the main components of the experimental pipeline. Some files are provided as templates rather than end-to-end reproduction scripts, because the original experiments used a mix of API-hosted models, locally hosted GPU models, provider-specific interfaces, and data artifacts that may be subject to safety, licensing, or platform-term constraints.



## Prompt Templates

The `prompts/` directory contains the main prompt templates used in the experiments. These files are provided separately so that the prompts can be inspected without searching through the code.

### Files

- `prompts/judge_censorship_score.txt`: prompt used by LLM judges to assign a censorship score to a prompt-response pair.
- `prompts/judge_censorship_score_multilingual.txt`: multilingual variant of the censorship-scoring prompt. This version instructs the judge to evaluate the response in the language it is written in and not penalize translation quality, grammar, fluency, or style.
- `prompts/translation_prompt.txt`: prompt template used to translate prompts or responses into a target language while preserving meaning, refusals, hedging, safety language, and template-like wording.

Some scripts may also include these prompts inline for convenience. The files in this directory are included to make the prompt text easy to inspect.

## Data Artifacts

The `data/` directory contains releasable data artifacts associated with the experiments.

### Files

- `data/calibration_set.csv`: calibration prompt-response pairs with selected judge scores used for score normalization.
- `data/censorship_dataset.csv`: representative examples from the curated local-censorship prompt dataset, where release is appropriate.

### Expected Dataset Columns

The curated prompt dataset uses fields such as:

- `Prompt`: input prompt.
- `Category`: high-level topic/category label.
- `Sub_Category`: finer-grained topic label, when available.
- `Source`: source label or construction method, when releasable.

The calibration score file uses fields such as:

- `Prompt`: input prompt.
- `Response`: model response used for calibration.
- `judge_claude_score`: censorship score from one LLM judge.
- `judge_grok_score`: censorship score from one LLM judge.
- `judge_gpt54_score`: censorship score from one LLM judge.


## Generation Templates

The experiments used a mix of API-hosted models and locally hosted GPU models. Because providers expose different APIs and response formats, the original experiments used model-specific wrappers. To avoid releasing provider-specific endpoints, credentials, deployment names, local cache paths, or account-specific configuration, this artifact provides anonymized generation templates.

The generation templates illustrate the common interface used across models and the normalized output format consumed by the downstream scoring and analysis code.

### Files

- `generation/api_model_wrapper_template.py`: template for OpenAI-compatible API-hosted models.
- `generation/gpu_model_wrapper_template.py`: template for locally hosted Hugging Face models.
- `generation/generate_dataset_template.py`: template for generation for the whole dataset

### Normalized Interface

Each model wrapper returns a dictionary with the following fields:

- `text`: final model response used for censorship scoring.
- `thinking`: reasoning trace, if available; otherwise an empty string.
- `usage`: token-usage metadata, when available.
- `raw_text`: raw generated text before parsing, when applicable.

Models that do not expose reasoning traces use an empty string for `thinking`.



### Dataset-Level Generation

The script `generation/generate_dataset_template.py` illustrates how model wrappers were applied to a prompt dataset. The same dataset-level generation logic was used for both API-hosted models and locally hosted GPU models. The only model-specific component is the wrapper function, which normalizes each model's output into a common dictionary format.

The script supports:

- loading an input prompt dataset;
- selecting a row range with `--start_idx` and `--end_idx`;
- resuming from an existing output file;
- retrying failed generations;
- periodically saving intermediate results;
- storing final responses and reasoning traces in separate columns.

Example usage:

```bash
python generation/generate_dataset_template.py \
  --input_file data/input_prompts.pkl \
  --output_file outputs/model_outputs.pkl \
  --prompt_col Prompt \
  --wrapper_module api_model_wrapper_template \
  --wrapper_function call_api_model \
  --start_idx 0 \
  --end_idx 100

```

## Censorship Analysis Framework

The `censorship framework/` directory contains the core code templates for computing the censorship metrics used in the paper. The analysis assumes that model responses have already been generated and scored by LLM judges. Each judge-scored model file should contain the prompt text and judge-score columns.

### Files

- `censorship framework/framework_utils.py`: shared utility functions for loading datasets, calibrating judge scores, building prompt-by-model score matrices, and computing global, model-level local, and group-level censorship metrics.
- `censorship framework/run_global_censorship_template.py`: template for computing global censorship relative to an uncensored baseline.
- `censorship framework/run_local_censorship_template.py`: template for computing model-level local censorship by comparing a target model against a reference set.
- `censorship framework/run_group_censorship_template.py`: template for computing group-level censorship by comparing a target group of models against a reference group.
- `censorship framework/plots/`: optional plotting templates for visualizing framework outputs.

### Inputs

The analysis templates expect judge-scored model files with columns such as:

- `Prompt`: the input prompt.
- `judge_claude_score`: censorship score assigned by one judge.
- `judge_gpt54_score`: censorship score assigned by one judge.
- `judge_grok_score`: censorship score assigned by one judge.
- Optional metadata columns such as `Category` and `Sub_Category`.

The exact judge-score column names can be configured through command-line arguments.

### Output

The scripts produce CSV files containing censorship metrics such as:

- mean censorship elevation or local gap;
- fraction of prompts with positive elevation/gap;
- Wilcoxon signed-rank test p-values;
- equivalence-test p-values;
- confidence intervals;
- final global/local/group-level censorship labels.

### Notes

The released scripts use anonymized model names and relative/example paths. The original experiments used the same analysis structure with model-specific score files produced by the generation and judge-scoring pipeline. Plotting scripts are provided as optional templates and are not required to run the core analysis.

