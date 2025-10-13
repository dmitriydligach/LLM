#!/usr/bin/env python3

import os, json, argparse, requests, pdfminer
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text

def resolve_prompt(input_text: str) -> str:
  """Replace a resource (file or URL) with its text"""

  start = input_text.find('[')
  end = input_text.find(']')

  if start == -1 or end == -1:
    return input_text

  file_path_or_url = input_text[start+1:end]

  if file_path_or_url.startswith('http'):
    text = get_page_text(file_path_or_url)
  elif file_path_or_url.endswith('.pdf'):
    text = extract_text(file_path_or_url)
  elif os.path.isfile(file_path_or_url):
    text = open(file_path_or_url).read()
  else:
    return input_text

  return input_text[:start] + '\n\n' + text + '\n' + input_text[end+1:]

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

  # text = '[/home/dima/Data/MimicIII/Discharge/Text/160090_discharge.txt]. Summarize!'
  # text = '[https://www.dmitriydligach.com/research]. Summarize the main points!'
  text = 'Summarize this text: [/home/dima/Data/Pcori/sample_protocol/NCT02938442/Prot_SAP_000.pdf].'
  expanded_text = resolve_prompt(text)
  print(expanded_text)

