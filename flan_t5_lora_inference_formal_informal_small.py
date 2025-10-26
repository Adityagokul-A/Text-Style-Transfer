import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# --- 1. Setup ---
# Base model name
model_name = "google/flan-t5-base"

# Path to the saved LoRA adapter from training
lora_adapter_path = "./flan-t5-base-lora-style-transfer-formal-informal-small/final_lora_adapter"

# Check if GPU is available
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
def generate_formal_sentence(informal_sentence, max_length=128):
    # Tokenize input
    inputs = tokenizer(informal_sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5, # beam search for better quality
            early_stopping=True
        )
    
    # Decode the generated tokens
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 4. Interactive Loop ---
print("\nType an informal sentence to convert to formal. Type 'exit' to quit.\n")

while True:
    informal_input = input("Informal: ").strip()
    if informal_input.lower() == "exit":
        print("Exiting...")
        break
    
    formal_output = generate_formal_sentence(informal_input)
    print(f"Formal : {formal_output}\n")
