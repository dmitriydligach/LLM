#!/usr/bin/env python3

"""
The prompt may contain a file, e.g.
[/home/dima/Data/MimicIII/Discharge/Text/160090_discharge.txt]. Summarize!
"""

import transformers, torch, os, json, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main(settings_file):
  """Chat with Llama"""

  settings = read_json_file(settings_file)

  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type= 'nf4')

  tokenizer = AutoTokenizer.from_pretrained(settings['model_path'])

  model = AutoModelForCausalLM.from_pretrained(
    settings['model_path'],
    quantization_config=quant_config,
    device_map=settings['device_map'])

  generator = transformers.pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map=settings['device_map'],
    pad_token_id=tokenizer.eos_token_id)

  conversation = [{'role': 'system', 'content': settings['sys_prompt']}]

  while True:
    user_input = input('\n>>> ')
    user_input = expand_prompt(user_input)
    conversation.append({'role': 'user', 'content': user_input})

    output = generator(
      conversation,
      do_sample=settings['do_sample'],
      temperature=settings['temperature'],
      top_p=settings['top_p'],
      max_new_tokens=settings['max_new_tokens'])

    conversation = output[0]['generated_text']
    print('\n' + conversation[-1]['content'])

def expand_prompt(input_text: str) -> str:
  """Replace possible path to a file with the file"""

  start = input_text.find('[')
  end = input_text.find(']')

  if start == -1 or end == -1:
    return input_text

  file_path = input_text[start+1:end]
  file_content = open(file_path).read()

  return input_text[:start] + '\n\n' + file_content + '\n' + input_text[end+1:]

def read_json_file(settings_json_file):
  """Read generation and other parameters"""

  with open(settings_json_file, 'r') as file:
    data = json.load(file)
  return data

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--settings',
    type=str,
    help='LLM configuration file',
    default='settings.json')
  args = parser.parse_args()

  main(args.config_file)
