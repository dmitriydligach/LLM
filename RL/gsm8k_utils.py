import re

MODEL_ID = "/home1/shared/Models/Llama-3.2-1B-Instruct"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"

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