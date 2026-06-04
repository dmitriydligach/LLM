# RL Project — GRPO Fine-tuning

## What this is
Reinforcement learning (GRPO) fine-tuning of `Qwen/Qwen2-0.5B-Instruct` using the TRL library.
Based on: https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl

## Files
- `grpo.py` — original script using **NuminaMath-TIR** dataset (complex competition math)
- `grpo_gsm8k.py` — variant using **GSM8K** dataset (simple grade-school word problems, easier to monitor)

## Architecture
- LoRA adapters (r=8, lora_alpha=32) on q_proj + v_proj
- Two reward functions: `format_reward` (checks `<think>...</think><answer>...</answer>` structure) and `accuracy_reward` (checks numerical answer correctness)
- Model outputs saved as LoRA adapter weights, reloaded with `PeftModel.from_pretrained`

## grpo_gsm8k.py key details
- Dataset: `openai/gsm8k` (main config), 50% split
- GSM8K answers are after `####` in the raw data — `extract_answer()` parses this
- Comma normalization in `accuracy_reward` (e.g. "1,234" → "1234")
- `max_completion_length=512` (needs room for multi-step reasoning)
- Output dir: `Qwen2-0.5B-GRPO-GSM8K`

## grpo.py key details
- Dataset: `AI-MO/NuminaMath-TIR`, 5% split
- Uses `math_verify` library for LaTeX answer verification
- Output dir: `Qwen2-0.5B-GRPO-test`

## Status
- `grpo_gsm8k.py` was just created and reviewed — ready to run for the first time
- `grpo.py` has been run previously (checkpoints may exist in `Qwen2-0.5B-GRPO-test/`)
