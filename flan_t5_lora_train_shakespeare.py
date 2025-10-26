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

# --- 1. Load Shakespeare Dataset ---
try:
    with open("shakespeare.json", "r", encoding="utf-8") as f:
        data = json.load(f)  # list of dicts with "modern_sentence" and "shakespeare_sentence"
        print(f"Loaded {len(data)} examples from shakespeare.json")
except FileNotFoundError:
    raise FileNotFoundError("❌ Could not find shakespeare.json. Run the conversion script first!")

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
    target_modules=["q", "k", "v", "o"]  # <-- CHANGED: Expanded targets
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 4. Data Preprocessing ---
def preprocess_function(examples):
    # <-- CHANGED: Added prompt prefix
    prefix = "Translate this text from Modern English to Shakespearean English: "
    inputs = [prefix + doc for doc in examples["modern_sentence"]]

    model_inputs = tokenizer(
        inputs,  # Use new inputs with prefix
        max_length=128,  # <-- CHANGED: Increased max_length
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=examples["shakespeare_sentence"],
        max_length=128,  # <-- CHANGED: Increased max_length
        truncation=True,
        padding="max_length"
    )

    # Replace padding token ids in labels with -100 so they are ignored in loss
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# --- 5. Sanity Check ---
print("\n--- Inspecting a preprocessed example ---")
example = tokenized_dataset[0]
print(f"Input: {tokenizer.decode(example['input_ids'], skip_special_tokens=True)}")
print(f"Target: {tokenizer.decode([x for x in example['labels'] if x != -100], skip_special_tokens=True)}")
print("-----------------------------------------\n")

# --- 6. Training Setup ---
output_dir = "./flan-t5-base-lora-shakespeare"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=15,
    learning_rate=3e-4,  # <-- CRITICAL CHANGE: Increased learning rate
    fp16=False,
    max_grad_norm=1.0,
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
        if logs and "loss" in logs:
            self.loss_history.append({"step": state.global_step, "loss": logs["loss"]})
            with open(self.log_file, "w") as f:
                json.dump(self.loss_history, f, indent=4)
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

# --- 8. Start Fine-tuning ---
print("Starting fine-tuning on Modern → Shakespeare dataset...")
trainer.train()

# --- 9. Save LoRA Adapter ---
final_adapter_path = os.path.join(output_dir, "final_lora_adapter")
trainer.save_model(final_adapter_path)

print("\n✅ Fine-tuning complete!")
print(f"Adapter saved to: {final_adapter_path}")
print(f"Logs in: {output_dir}")
print(f"Training loss history: {loss_log_file}")