#!/usr/bin/env python3

"""
The prompt may contain a file, e.g.
[/home/dima/Data/MimicIII/Discharge/Text/160090_discharge.txt]. Summarize!
"""

import os, argparse, utils
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
    device_map=settings['device_map'])
  train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

  training_args = DPOConfig(output_dir="DPOModel")
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
