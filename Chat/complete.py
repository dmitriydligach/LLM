#!/usr/bin/env python3

"""
Interact with a base (non-instruct) model: each prompt is continued as plain text.
"""

import transformers, torch, os, argparse, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Lib import utils
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

transformers.logging.set_verbosity_error()

def main(settings_file):
  """Complete prompts with a base model"""

  settings = utils.read_yaml_file(settings_file)

  tokenizer = AutoTokenizer.from_pretrained(settings['model_path'])
  model = AutoModelForCausalLM.from_pretrained(
    settings['model_path'],
    dtype=torch.bfloat16,
    device_map=settings['device_map'])

  generator = transformers.pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer)

  gen_config = GenerationConfig(
    do_sample=settings['do_sample'],
    temperature=settings['temperature'],
    top_p=settings['top_p'],
    max_new_tokens=settings['max_new_tokens'],
    pad_token_id=tokenizer.eos_token_id)

  while True:
    prompt = input('\n>>> ')
    output = generator(prompt, generation_config=gen_config, return_full_text=False)
    print('\n' + output[0]['generated_text'])

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--config-file',
    type=str,
    help='LLM configuration file',
    default='base.yaml')
  args = parser.parse_args()

  main(args.config_file)
