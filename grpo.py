import re
import time
import torch
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

#
# Based on https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl
#

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DATASET_ID = "AI-MO/NuminaMath-TIR"
OUTPUT_DIR = "Qwen2-0.5B-GRPO-test"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>")

# ==========================================
# 2. LOAD & PREPARE DATASET
# ==========================================
print(f"Loading dataset: {DATASET_ID}...")
# Loading only 5% slice as per the cookbook workflow
train_dataset, test_dataset = load_dataset(DATASET_ID, split=['train[:5%]', 'test[:5%]'])


def make_conversation(example):
    return {"prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},],}


print("Mapping chat structures...")
train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

# Clean out unused structural columns (keeping 'solution' and 'prompt')
train_dataset = train_dataset.remove_columns(['messages', 'problem'])

# ==========================================
# 3. INITIALIZE MODEL & CONFIG LORA
# ==========================================
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

# ==========================================
# 4. DEFINE REWARD FUNCTIONS
# ==========================================
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])

        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)

    return rewards

# ==========================================
# 5. GRPO TRAINING CONFIGURATION
# ==========================================
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    remove_unused_columns=False,  # Vital context to keep 'solution' column for accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    max_completion_length=64,  # Default: 256 (Kept small for resource optimization)
    num_generations=4,  # Default: 8
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=False,  # Modified to False for seamless local scripting
    save_strategy="steps",
    save_steps=10,
)

# ==========================================
# 6. INITIATE TRAINING
# ==========================================
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

# ==========================================
# 7. PERFORMANCE EVALUATION
# ==========================================
print("\n--- Evaluating Trained Model Performance ---")
trained_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Reload the locally trained model adapter weights
# trained_model = AutoModelForCausalLM.from_pretrained(
#     OUTPUT_DIR,
#     torch_dtype="auto",
#     device_map="auto",
# )

# claude tells me that this is the way to go
# otherwise not obiouvs if the adapter is merged with the base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto")
trained_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

def generate_with_reasoning(prompt):
    """Utility function to measure generation lengths and latency."""

    # Build the linear prompt string from the dataset schema
    # prompt_text = " ".join(entry['content'] for entry in prompt)
    # inputs = trained_tokenizer(prompt_text, return_tensors="pt").to(trained_model.device)

    # claude didn't like the two lines above and suggest the lines below instead
    prompt_text = trained_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = trained_tokenizer(prompt_text, return_tensors="pt").to(trained_model.device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = trained_model.generate(**inputs, max_length=500)
    end_time = time.time()

    generated_text = trained_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    inference_duration = end_time - start_time

    num_input_tokens = inputs['input_ids'].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, inference_duration, num_generated_tokens


# Evaluate on the first sample of our test dataset split
test_prompt = test_dataset['prompt'][3]
generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(test_prompt)

print("\n--- Generated Output ---")
print(generated_text)
print("\n--- Inference Statistics ---")
print(f"Inference time: {inference_duration:.2f} seconds")
print(f"Generated tokens: {num_generated_tokens}")
