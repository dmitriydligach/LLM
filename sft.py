#!/usr/bin/env python3

import os, argparse, utils, torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datasets import Dataset

def make_sft_dataset(tokenizer, n: int = 25) -> Dataset:
  qa_pairs = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is 2 + 2?", "4"),
    ("What is 15 * 3?", "45"),
    ("What is the square root of 81?", "9"),
    ("What is the chemical formula for water?", "H2O"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("Who wrote '1984'?", "George Orwell"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("How many continents are there on Earth?", "7"),
    ("How many minutes are in an hour?", "60"),
    ("How many seconds are in a minute?", "60"),
    ("How many days are in a leap year?", "366"),
    ("What is the first element on the periodic table?", "Hydrogen"),
    ("What is the smallest prime number?", "2"),
    ("What is 5 factorial (5!)?", "120"),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
    ("What does HTML stand for?", "HyperText Markup Language"),
    ("What is 0°C in Fahrenheit?", "32°F"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Brazil?", "Brasília"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("How many sides does a hexagon have?", "6"),
  ][:n]

  rows = []
  for q, a in qa_pairs:
    messages = [
      {"role": "user", "content": q},
      {"role": "assistant", "content": "Hello! " + a},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    rows.append({"text": text})

  return Dataset.from_list(rows)

def main(settings_file):
  settings = utils.read_json_file(settings_file)

  tokenizer = AutoTokenizer.from_pretrained(settings["model_path"])

  model = AutoModelForCausalLM.from_pretrained(
    settings["model_path"],
    dtype=torch.bfloat16,
  )

  train_dataset = make_sft_dataset(tokenizer)

  training_args = SFTConfig(
    output_dir="SFT",
    per_device_train_batch_size=16,
    max_length=512,          # SFT uses max_seq_length rather than max_prompt_length/max_length
    learning_rate=1e-5,
    max_steps=10,                # train by optimizer steps (like your current script)
    logging_steps=1,
    logging_strategy="steps",
    bf16=True,
    dataset_text_field="text"
  )

  trainer = SFTTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset)

  trainer.train()
  trainer.save_model("SFT")
  tokenizer.save_pretrained("SFT")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--settings",
    type=str,
    help="LLM configuration file",
    default="settings.json",
  )
  args = parser.parse_args()
  main(args.settings)
