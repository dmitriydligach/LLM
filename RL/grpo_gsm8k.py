import re
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

#
# Based on https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl
# Modified to use GSM8K dataset for easier interpretability
#

MODEL_ID = "/home1/shared/Models/Llama-3.2-1B-Instruct"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
OUTPUT_DIR = "Model"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a math question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> numerical answer here </answer>")

print(f"Loading dataset: {DATASET_ID}...")
train_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split='train')

def extract_answer(answer_text):
    """Extract the numerical answer from GSM8K format: 'reasoning ... #### answer'"""

    match = re.search(r'####\s*(.*?)(?:\n|$)', answer_text)
    if match:
        return match.group(1).strip().replace(',', '')

    return answer_text.strip().replace(',', '')

def make_conversation(example):
    """Chat template stuff"""

    return {"prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},],
            "answer": extract_answer(example["answer"])}

train_dataset = train_dataset.map(make_conversation)

# Clean out unused structural columns (keeping 'answer' and 'prompt')
train_dataset = train_dataset.remove_columns(['question'])

print(f"Loading baseline model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],)

print("Applying LoRA parameters...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has the required <think>/<answer> format."""

    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]

    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion matches the ground truth answer."""

    answers = kwargs['answer']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, ground_truth in zip(completion_contents, answers):
        # Strip commas (e.g. "1,234" -> "1234") before comparing
        # Try to find the answer inside <answer> tags first, fall back to last number in response
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        search_text = answer_match.group(1) if answer_match else content
        numbers = re.findall(r'-?\d+(?:\.\d+)?', search_text.replace(',', ''))

        if numbers:
            predicted_answer = numbers[-1]  # Take the last number
            if predicted_answer.strip() == ground_truth.strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    remove_unused_columns=False,  # Vital context to keep 'answer' column for accuracy_reward
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    bf16=True,
    max_completion_length=256,
    num_generations=8,
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=False,  # Modified to False for seamless local scripting
    save_strategy="steps",
    save_steps=10)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset
)

print("Starting GRPO training pass...")
trainer.train()

print(f"Saving fine-tuned model checkpoint to {OUTPUT_DIR}...")
trainer.save_model(training_args.output_dir)
