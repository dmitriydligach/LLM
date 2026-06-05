import re
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "/home1/shared/Models/Llama-3.2-1B-Instruct"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
TRAINED_MODEL_DIR = "Model"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a math question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> numerical answer here </answer>")

def extract_ground_truth(answer_text):
    """Extract the numerical answer from GSM8K format: 'reasoning ... #### answer'"""

    match = re.search(r'####\s*(.*?)(?:\n|$)', answer_text)
    if match:
        return match.group(1).strip().replace(',', '')

    return answer_text.strip().replace(',', '')

def make_conversation(example):
    return {"prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}],
            "answer": extract_ground_truth(example["answer"])}

def extract_predicted_answer(generated_text):
    """Extract numerical answer from model output, preferring <answer> tags."""

    answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
    search_text = answer_match.group(1) if answer_match else generated_text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', search_text.replace(',', ''))

    return numbers[-1] if numbers else None

def evaluate(model, tokenizer, dataset):
    """Run greedy decoding on the full dataset and return accuracy."""

    model.eval()
    correct = 0

    for example in tqdm(dataset, desc="Evaluating"):
        prompt_text = tokenizer.apply_chat_template(
            example['prompt'], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted = extract_predicted_answer(generated_text)

        if predicted is not None and predicted.strip() == example['answer'].strip():
            correct += 1

    return correct / len(dataset)

print(f"Loading test set: {DATASET_ID}...")
test_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split='test')
test_dataset = test_dataset.map(make_conversation)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# --- Base model ---
print(f"\nEvaluating base model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
base_accuracy = evaluate(base_model, tokenizer, test_dataset)
del base_model
torch.cuda.empty_cache()

# --- Trained model ---
print(f"\nEvaluating trained model: {TRAINED_MODEL_DIR}")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
trained_model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_DIR)
trained_accuracy = evaluate(trained_model, tokenizer, test_dataset)

# --- Results ---
print("\n=== GSM8K Evaluation Results ===")
print(f"Base model:    {base_accuracy:.1%}  ({int(base_accuracy * len(test_dataset))}/{len(test_dataset)})")
print(f"Trained model: {trained_accuracy:.1%}  ({int(trained_accuracy * len(test_dataset))}/{len(test_dataset)})")
print(f"Improvement:   {trained_accuracy - base_accuracy:+.1%}")
print(f"\nReference (published): Llama-3.2-1B-Instruct ~44%")
