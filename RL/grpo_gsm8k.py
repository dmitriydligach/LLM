import re
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from gsm8k_utils import MODEL_ID, DATASET_ID, DATASET_CONFIG, make_conversation, extract_predicted_answer

#
# Based on https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl
# Modified to use GSM8K dataset for easier interpretability
#

OUTPUT_DIR = "Model"

print(f"Loading dataset: {DATASET_ID}...")
train_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split='train')
train_dataset = train_dataset.map(make_conversation)

# Clean out unused structural columns (keeping 'answer' and 'prompt')
train_dataset = train_dataset.remove_columns(['question'])

print(f"Loading baseline model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto")

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
        predicted_answer = extract_predicted_answer(content)

        if predicted_answer is not None and predicted_answer.strip() == ground_truth.strip():
            rewards.append(1.0)
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
