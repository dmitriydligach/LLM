#!/usr/bin/env python3

"""
The prompt may contain a file, e.g.
[/home/dima/Data/MimicIII/Discharge/Text/160090_discharge.txt]. Summarize!
"""

import logging
# hide PyTorch's harmless "Redirects are currently not supported in MacOs" note
logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').setLevel(logging.ERROR)

import transformers, torch, os, argparse, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Lib import utils
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from rich.console import Console
from rich.markdown import Markdown

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

transformers.logging.set_verbosity_error()

def main(settings_file):
  """Chat with Llama"""

  settings = utils.read_yaml_file(settings_file)

  tokenizer = AutoTokenizer.from_pretrained(settings['model_path'], clean_up_tokenization_spaces=False)

  quant_config = None
  if settings.get('quantize'):
    quant_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4')

  model = AutoModelForCausalLM.from_pretrained(
    settings['model_path'],
    quantization_config=quant_config,
    device_map=settings['device_map'])

  if settings.get('adapter_path'):
    print(f"Loading adapter: {settings['adapter_path']}")
    model = PeftModel.from_pretrained(model, settings['adapter_path'])

  generator = transformers.pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    dtype=torch.bfloat16,
    device_map=settings['device_map'])

  console = Console()

  conversation = [{'role': 'system', 'content': settings['sys_prompt']}]

  while True:
    user_input = input('\n>>> ')
    user_input = utils.resolve_prompt(user_input)
    conversation.append({'role': 'user', 'content': user_input})

    gen_config = GenerationConfig(
      do_sample=settings['do_sample'],
      temperature=settings['temperature'],
      top_p=settings['top_p'],
      top_k=settings.get('top_k'),
      max_new_tokens=settings['max_new_tokens'],
      pad_token_id=tokenizer.eos_token_id)

    template_kwargs = {}
    if 'enable_thinking' in settings:
      template_kwargs['enable_thinking'] = settings['enable_thinking']

    output = generator(conversation, generation_config=gen_config, tokenizer_encode_kwargs=template_kwargs)

    conversation = output[0]['generated_text']
    console.print()
    console.print(Markdown(conversation[-1]['content']))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--config-file',
    type=str,
    help='LLM configuration file',
    default='mac.yaml')
  args = parser.parse_args()

  main(args.config_file)
