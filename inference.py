import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def inference(args):
    # --- 1. Setup ---
    model_name = "google/flan-t5-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Tokenizer and Model with LoRA Adapter ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    model.to(device)
    model.eval()

    # --- 3. Inference Function ---
    def generate_text(text, max_length=128):
        source_lang, target_lang = args.style.split('_to_')
        input_text = f"Convert to <{target_lang}> from <{source_lang}>: {text}"
        
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- 4. Interactive Loop or Single Inference ---
    if args.input_text:
        output_text = generate_text(args.input_text)
        print(f"Input: {args.input_text}")
        print(f"Output: {output_text}")
    else:
        print(f"\nStarting interactive inference for style: {args.style}. Type 'exit' to quit.\n")
        while True:
            input_text = input("Input: ").strip()
            if input_text.lower() == "exit":
                print("Exiting...")
                break
            
            output_text = generate_text(input_text)
            print(f"Output: {output_text}\n")

def main():
    parser = argparse.ArgumentParser(description="Run inference for a text style transfer model.")
    parser.add_argument("--style", type=str, required=True, help="Style for the transfer (e.g., 'modern_to_shakespeare', 'informal_to_formal').")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the LoRA adapter for inference.")
    parser.add_argument("--input_text", type=str, help="A single input text for quick inference.")

    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    main()
