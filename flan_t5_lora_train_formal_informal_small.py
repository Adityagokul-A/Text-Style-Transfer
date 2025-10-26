
import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# --- 1. Define the Dataset ---
# Ensure you have a file named "formal_informal_small.json" in the same directory
try:
    with open("formal_informal_small.json", "r", encoding="utf-8") as f:
        data = json.load(f)  # list of dicts with "informal" and "formal"
        print(f"Loaded {len(data)} examples from formal_informal_small.json")
except FileNotFoundError:
    print("Error: formal_informal_small.json not found. Please create this file with your dataset.")
    # Creating a dummy dataset for demonstration if the file doesn't exist
    data = [
        {"informal": "hey how r u?", "formal": "Hello, how are you?"},
        {"informal": "wanna grab a bite?", "formal": "Would you like to get some food?"}
    ]
    print("Using a dummy dataset for demonstration purposes.")


dataset = Dataset.from_list(data)

# --- 2. Model and Tokenizer Setup ---
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- 3. LoRA Configuration ---
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]  # Target the query and value matrices for LoRA
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 4. Data Preprocessing ---
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["informal_sentence"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=examples["formal_sentence"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    # Replace padding token id's in labels with -100 so they are ignored in loss
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# --- 5. [NEW] Data Sanity Check ---
# This block helps verify that your data is processed correctly before training.
print("\n--- Inspecting a preprocessed example ---")
try:
    example = tokenized_dataset[0]
    print(f"Input IDs: {example['input_ids']}")
    print(f"Decoded Input: {tokenizer.decode(example['input_ids'], skip_special_tokens=True)}")
    print(f"\nLabels: {example['labels']}")

    # Filter out -100 to see what the actual target tokens are
    actual_labels = [label for label in example['labels'] if label != -100]
    print(f"Actual Label IDs (non-padding): {actual_labels}")
    print(f"Decoded Labels: {tokenizer.decode(actual_labels, skip_special_tokens=True)}")
    print("-----------------------------------------\n")

    if not actual_labels:
        raise ValueError("FATAL: The first example has no valid labels after preprocessing. "
                         "Check your data for empty target sentences or if max_length is too short.")
except IndexError:
    print("FATAL: The dataset is empty. Cannot proceed with training.")
    exit()


# --- 6. Training Arguments and Trainer Setup ---
output_dir = "./flan-t5-base-lora-style-transfer-formal-informal-small"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    # === STABILITY FIXES APPLIED HERE ===
    learning_rate=5e-5,          # 1. Lowered learning rate for more stable updates.
    fp16=False,                  # 2. Disabled fp16 for debugging. Re-enable later if training is stable.
    max_grad_norm=1.0,           # 3. Added gradient clipping to prevent exploding gradients.
    # ====================================
    logging_dir=os.path.join(output_dir, "logs"),
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    disable_tqdm=False
)

# --- 7. Custom Callback for Logging ---
class CustomLossCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # We only care about training loss
        if logs and "loss" in logs:
            self.loss_history.append({"step": state.global_step, "loss": logs["loss"]})
            try:
                with open(self.log_file, "w") as f:
                    json.dump(self.loss_history, f, indent=4)
            except IOError as e:
                print(f"Could not write to log file: {e}")
            # Also log grad_norm if available
            if "grad_norm" in logs:
                print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}, Grad Norm = {logs['grad_norm']:.4f}")
            else:
                print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")


loss_log_file = os.path.join(output_dir, "training_loss.json")
custom_callback = CustomLossCallback(loss_log_file)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[custom_callback]
)

# --- 8. Start Training ---
print("Starting fine-tuning with stability fixes...")
trainer.train()

# --- 9. Save Final LoRA Adapter ---
final_adapter_path = os.path.join(output_dir, "final_lora_adapter")
trainer.save_model(final_adapter_path)

print("\nFine-tuning complete! LoRA adapter saved.")
print(f"Final adapter saved in: {final_adapter_path}")
print(f"Checkpoints and logs saved in: {output_dir}")
print(f"Training loss history saved to: {loss_log_file}")
