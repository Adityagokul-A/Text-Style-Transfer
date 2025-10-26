import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from visual_embeddings_utils import visualize_embeddings
import textstat
import re
from collections import Counter

# Streamlit Page Configuration
st.set_page_config(page_title="Multi-Domain Style Transfer", layout="centered")

st.title("Multi-Domain Text Style Transfer")
st.write("Convert text between different styles — Medical ↔ Casual, Legal ↔ Casual, Informal ↔ Formal, and more.")


# Model Loader
@st.cache_resource
def load_model(adapter_name: str):
    """Load base FLAN-T5 and the LoRA adapter for the selected style."""
    base_model_name = "google/flan-t5-base"
    adapter_dir = os.path.join("adapters", adapter_name)

    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Adapter folder not found: {adapter_dir}")

    # AUTOMATIC DEVICE SELECTION: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Using device: **{device}** {'(GPU accelerated)' if device.type == 'cuda' else '(CPU mode)'}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

# Text Generation
def generate_text(model, tokenizer, device, text: str, max_length: int = 128, 
                  temperature: float = 0.7, top_p: float = 0.9, 
                  repetition_penalty: float = 1.2, num_beams: int = 5):
    """Generate transformed text using LoRA adapter."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI Controls
# ============================================================
input_text = st.text_area(
    "Enter your text:",
    placeholder="Type a sentence in any style...",
    height=15
)

# Dynamically list all adapters in ./adapters/
available_adapters = sorted(
    [d for d in os.listdir("adapters") if os.path.isdir(os.path.join("adapters", d))]
)

if not available_adapters:
    st.error("No adapters found in './adapters/'. Please add your LoRA adapter folders.")
    st.stop()

adapter_choice = st.selectbox("Select Style Transfer Adapter", available_adapters)

# Convert Button
if st.button("Convert", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to convert.")
    else:
        try:
            source, target = adapter_choice.split("_to_")
        except ValueError:
            source, target = "source", "target"

        formatted_input = f"Convert to <{target}> from <{source}>: {input_text}"

        with st.spinner(f"Loading model for '{adapter_choice}' ..."):
            tokenizer, model, device = load_model(adapter_choice)

        with st.spinner("Generating transformed text..."):
            output_text = generate_text(
                model, tokenizer, device, formatted_input)
            
            st.success("Conversion complete!")
            
            # Display output prominently
            st.markdown("---")
            st.subheader("Converted Text")
            st.markdown(f"```\n{output_text}\n```")
            
            # store these for later reruns
            st.session_state["last_input"] = input_text
            st.session_state["last_output"] = output_text
            st.session_state["adapter_choice"] = adapter_choice
            st.session_state["tokenizer"] = tokenizer
            st.session_state["model"] = model
            st.session_state["device"] = device

# Visualization (persists after conversion)
if "last_output" in st.session_state:
    st.markdown("---")

    # Embedding Visualization Section
    st.subheader("Embedding Space Visualization")
    show_embeddings = st.checkbox("Show sentence embeddings", value=False)
    
    if show_embeddings:
        with st.spinner("Computing embeddings... this may take a few seconds."):
            try:
                fig = visualize_embeddings(
                    [(st.session_state["last_input"], st.session_state["last_output"])],
                    st.session_state["model"],
                    st.session_state["tokenizer"],
                    st.session_state["device"],
                    title=f"Embedding shift: {st.session_state['adapter_choice']}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("**Tip**: The arrows show how the model transforms the semantic representation of your text in high-dimensional space.")
            except Exception as e:
                st.error(f"Could not generate embedding visualization: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, HuggingFace Transformers, and PEFT (LoRA)")
