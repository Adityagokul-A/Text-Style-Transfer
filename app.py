import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="Multi-Domain Style Transfer", layout="centered")

# --- Title & Description ---
st.title("Multi-Domain Text Style Transfer")
st.write("Convert text between different styles (Medical ↔ Casual, Legal ↔ Casual,Casual ↔ Formal etc.).")

# --- User Input ---
input_text = st.text_area(
    "Enter your text here:",
    placeholder="Type something casual, scientific, sarcastic, etc."
)

# --- Style Selection ---
col1, col2 = st.columns(2)

with col1:
    source_style = st.selectbox(
        "Source Style",
        ["casual", "formal", "legal", "medical"]
    )

with col2:
    target_style = st.selectbox(
        "Target Style",
        ["formal", "casual", "legal", "medical"]
    )

# --- Convert Button ---
if st.button("Convert"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Converting..."):
            # Placeholder for backend call (to be added later)
            st.success("Done!")
            st.subheader("Converted Text")
            st.write("This is a placeholder output. The model will generate text here once integrated.")