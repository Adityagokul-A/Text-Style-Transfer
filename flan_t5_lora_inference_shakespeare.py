import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

print(os.getcwd())

# --- 1. Setup ---
# Base model
model_name = "google/flan-t5-base"

# Path to the saved LoRA adapter (update this if you saved it elsewhere)
lora_adapter_path = "./flan-t5-base-lora-shakespeare/final_lora_adapter"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load Tokenizer and Model with LoRA Adapter ---
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load LoRA adapter on top of base model
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.to(device)
model.eval()

# --- 3. Inference Function ---
def translate_to_shakespeare(modern_sentence, max_length=64):
    """Convert modern English to Shakespearean style using fine-tuned LoRA model."""
    # Add instruction prefix (important for T5-style models)
    input_text = f"Convert to <Shakespeare> from <modern>: {modern_sentence}"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)

    # Generate Shakespearean text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,       # Beam search improves output quality
            early_stopping=True
        )

    # Decode generated output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 4. Interactive Loop ---
print("\nType a modern English sentence to translate into Shakespearean English.")
print("Type 'exit' to quit.\n")

while True:
    modern_input = input("Modern: ").strip()
    if modern_input.lower() == "exit":
        print("Exiting...")
        break

    shakespeare_output = translate_to_shakespeare(modern_input)
    print(f"Shakespeare: {shakespeare_output}\n")
