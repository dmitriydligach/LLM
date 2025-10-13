#!/usr/bin/env python3

import transformers, torch, os, json, argparse, requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bs4 import BeautifulSoup

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

def get_page_text(url: str) -> str:
  """Fetch a web page and return its visible text"""

  response = requests.get(url, timeout=10)
  response.raise_for_status()  # will raise an HTTPError if not 200 OK
  soup = BeautifulSoup(response.text, "html.parser")

  # Remove non-content tags
  for tag in soup(["script", "style", "noscript"]):
    tag.decompose()

  # Extract and normalize visible text
  return soup.get_text(separator='\n', strip=True)

if __name__ == "__main__":

  url = 'https://www.dmitriydligach.com/research'
  txt = get_page_text(url)
  print(txt)

