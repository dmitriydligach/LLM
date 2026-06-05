import re
import time
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Todo:
#
# Add a formal evaluation at the end using the official scoring metric for this dataset
# Add token count to the sample examples
#

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
train_dataset, test_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=['train[:50%]', 'test[:50%]'])

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
test_dataset = test_dataset.map(make_conversation)

# Clean out unused structural columns (keeping 'answer' and 'prompt')
train_dataset = train_dataset.remove_columns(['question'])
test_dataset = test_dataset.remove_columns(['question'])

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
    target_modules=["q_proj", "v_proj"],)

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
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    max_completion_length=512,
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

print("\n--- Evaluating Trained Model Performance ---")
trained_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Reload the locally trained model adapter weights
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto")
trained_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

def generate_with_reasoning(prompt):
    """Utility function to measure generation lengths and latency."""

    prompt_text = trained_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = trained_tokenizer(prompt_text, return_tensors="pt").to(trained_model.device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = trained_model.generate(**inputs, max_new_tokens=512)
    end_time = time.time()

    generated_text = trained_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    inference_duration = end_time - start_time

    num_input_tokens = inputs['input_ids'].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, inference_duration, num_generated_tokens

# Evaluate on a few samples from our test dataset split
print("\n--- Evaluating on test samples ---")
num_samples = min(3, len(test_dataset))
for idx in range(num_samples):
    test_prompt = test_dataset['prompt'][idx]
    test_answer = test_dataset['answer'][idx]
    generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(test_prompt)

    print(f"\n--- Sample {idx + 1} ---")
    print(f"Expected answer: {test_answer}")
    print(f"Generated output:\n{generated_text}")
    print(f"Inference time: {inference_duration:.2f} seconds")
    print(f"Generated tokens: {num_generated_tokens}")
