#!/usr/bin/env python3

import transformers, torch, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_path1 = '/home1/shared/Models/Llama3.1/Llama-3.1-8B-Instruct'
model_path2 = '/home1/shared/Models/Llama3.1/Llama-3.1-8B-Instruct'

do_sample = True
temperature = 1.0
top_p = 0.9
max_new_tokens = 5000

device_map1 = {'': 'cuda:0'}
device_map2 = {'': 'cuda:1'}

init_prompt = 'What is the meaning of life?'

sys_prompt1 = "Your name is A. You read what your conversation partner (whose name is B) " \
              "says and you comment on it. Your comments are precise and to the point."
sys_prompt2 = "Your name is B. You read what your conversation partner (whose name is A) " \
              "says and you comment on it. Your comments are precise and to the point. " \
              "You remind A what this conversation is about."

def main():
  """Ask for input and feed into llama2"""

  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type= 'nf4')

  tokenizer1 = AutoTokenizer.from_pretrained(model_path1)
  tokenizer2 = AutoTokenizer.from_pretrained(model_path2)

  model1 = AutoModelForCausalLM.from_pretrained(
    model_path1,
    quantization_config=quant_config,
    device_map=device_map1)
  generator1 = transformers.pipeline(
    task='text-generation',
    model=model1,
    tokenizer=tokenizer1,
    torch_dtype=torch.bfloat16,
    device_map=device_map1,
    pad_token_id=tokenizer1.eos_token_id)

  model2 = AutoModelForCausalLM.from_pretrained(
    model_path2,
    quantization_config=quant_config,
    device_map=device_map2)
  generator2 = transformers.pipeline(
    task='text-generation',
    model=model2,
    tokenizer=tokenizer2,
    torch_dtype=torch.bfloat16,
    device_map=device_map2,
    pad_token_id=tokenizer2.eos_token_id)

  conversation1 = [{'role': 'system', 'content': sys_prompt1}]
  conversation2 = [{'role': 'system', 'content': sys_prompt2}]

  conversation1.append({'role': 'user', 'content': init_prompt})
  conversation2.append({'role': 'user', 'content': init_prompt})

  while True:
    output1 = generator1(
      conversation1,
      do_sample=do_sample,
      temperature=temperature,
      top_p=top_p,
      max_new_tokens=max_new_tokens)

    conversation1 = output1[0]['generated_text']
    print('\nA: ' + conversation1[-1]['content'])

    conversation2.append({'role': 'user', 'content': conversation1[-1]['content']})

    output2 = generator2(
      conversation2,
      do_sample=do_sample,
      temperature=temperature,
      top_p=top_p,
      max_new_tokens=max_new_tokens)

    conversation2 = output2[0]['generated_text']
    print('\nB: ' + conversation2[-1]['content'])

    conversation1.append({'role': 'user', 'content': conversation2[-1]['content']})

if __name__ == "__main__":

  main()
