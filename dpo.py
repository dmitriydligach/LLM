#!/usr/bin/env python3

"""
The prompt may contain a file, e.g.
[/home/dima/Data/MimicIII/Discharge/Text/160090_discharge.txt]. Summarize!
"""

import os, argparse, utils, torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main(settings_file):
  """Chat with Llama"""

  settings = utils.read_json_file(settings_file)

  tokenizer = AutoTokenizer.from_pretrained(settings['model_path'])

  model = AutoModelForCausalLM.from_pretrained(
    settings['model_path'],
    dtype=torch.bfloat16)

  train_dataset = load_dataset(
    path="trl-lib/ultrafeedback_binarized",
    split="train").shuffle(seed=42).select(range(25))

  training_args = DPOConfig(
    output_dir="DPOModel",
    per_device_train_batch_size=16,
    # gradient_accumulation_steps=8,
    bf16=True,
    max_prompt_length=256,
    max_length=512,
    num_train_epochs=1,
    precompute_ref_log_probs=True)

  trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset)

  trainer.train()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--settings',
    type=str,
    help='LLM configuration file',
    default='settings.json')
  args = parser.parse_args()

  main(args.settings)
