import os
import json
import torch
import argparse
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

def train(args):
    # --- 1. Load Dataset ---
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples from {args.dataset_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {args.dataset_path}.")

    dataset = Dataset.from_list(data)

    # --- 2. Model and Tokenizer Setup ---
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # --- 3. LoRA Configuration ---
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Data Preprocessing ---
    def preprocess_function(examples):
        source_lang, target_lang = args.style.split('_to_')
        
        inputs = [f"Convert to <{target_lang}> from <{source_lang}>: {ex}" for ex in examples[f"{source_lang}_sentence"]]

        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            text_target=examples[f"{target_lang}_sentence"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        labels["input_ids"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in seq]
            for seq in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # --- 5. Data Sanity Check ---
    print("\n--- Inspecting a preprocessed example ---")
    try:
        example = tokenized_dataset[0]
        print(f"Decoded Input: {tokenizer.decode(example['input_ids'], skip_special_tokens=True)}")
        actual_labels = [label for label in example['labels'] if label != -100]
        print(f"Decoded Labels: {tokenizer.decode(actual_labels, skip_special_tokens=True)}")
        print("-----------------------------------------\n")
    except IndexError:
        print("WARN: Could not display sanity check for an empty dataset.")


    # --- 6. Training Arguments and Trainer Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=False,
        max_grad_norm=1.0,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        disable_tqdm=False
    )

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

    loss_log_file = os.path.join(args.output_dir, "training_loss.json")
    custom_callback = CustomLossCallback(loss_log_file)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[custom_callback]
    )

    # --- 7. Start Training ---
    print(f"Starting fine-tuning for style: {args.style}...")
    trainer.train()

    # --- 8. Save Final LoRA Adapter ---
    final_adapter_path = os.path.join(args.output_dir, "final_lora_adapter")
    trainer.save_model(final_adapter_path)

    print("\n✅ Fine-tuning complete!")
    print(f"Adapter saved to: {final_adapter_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a text style transfer model.")
    
    # Core arguments
    parser.add_argument("--style", type=str, required=True, help="Style for the transfer (e.g., 'modern_to_shakespeare', 'informal_to_formal').")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset (JSON file).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")

    # Hyperparameter arguments
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability.")
    parser.add_argument("--lora_target_modules", nargs='+', default=['q', 'v'], help="LoRA target modules (e.g., q v k o).")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
