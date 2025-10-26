import json
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from tqdm import tqdm

# --- 1. Configuration ---
MODEL_NAME = "google/flan-t5-base"
LORA_ADAPTER_PATH = "./flan-t5-base-lora-shakespeare/final_lora_adapter"
DATASET_PATH = "shakespeare.json"
SAMPLE_SIZE = 100  # Number of sentence pairs to visualize. Reduced slightly for clearer arrows.
RANDOM_SEED = 42
ARROW_SIZE_SCALE = 0.05 # Adjust this to change the size of the arrowheads relative to the plot scale

# Set seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 2. Model & Data Loading ---
print("Loading model and tokenizer...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model.to(DEVICE)
model.eval()

print("Loading and sampling dataset...")
try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if len(data) < SAMPLE_SIZE:
            print(f"Warning: Sample size ({SAMPLE_SIZE}) is larger than dataset ({len(data)}). Using all data.")
            data_sample = data
            SAMPLE_SIZE = len(data)
        else:
            data_sample = random.sample(data, SAMPLE_SIZE)
    print(f"Loaded {len(data)} pairs, using {SAMPLE_SIZE} for visualization.")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {DATASET_PATH}. Make sure it's in the same directory.")
    exit()

# --- 3. Embedding Function (Encoder + Masked Mean Pooling) ---
def get_embedding(text, model, tokenizer, device):
    """
    Gets a single vector embedding for a text by taking the
    mean of its encoder's last hidden state, ignoring padding.
    """
    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        ).to(device)

        # Get encoder output
        encoder_outputs = model.base_model.encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

        # Get last hidden state: [batch_size, seq_len, hidden_dim]
        last_hidden_state = encoder_outputs.last_hidden_state

        # --- Masked Mean Pooling ---
        # Get attention mask: [batch_size, seq_len]
        attention_mask = inputs.attention_mask

        # Expand mask to match hidden state: [batch_size, seq_len, hidden_dim]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)

        # Zero out padding tokens
        masked_states = last_hidden_state * mask_expanded

        # Sum non-padding tokens: [batch_size, hidden_dim]
        summed_states = torch.sum(masked_states, dim=1)

        # Count non-padding tokens: [batch_size, 1]
        token_counts = torch.sum(attention_mask, dim=1).unsqueeze(-1)

        # Handle potential division by zero if a text somehow tokenizes to only special tokens
        token_counts = torch.clamp(token_counts, min=1)

        # Calculate mean
        mean_pooled_embedding = summed_states / token_counts

        # Return as a numpy array, removing the batch dimension
        return mean_pooled_embedding.squeeze().cpu().numpy()

# --- 4. Generate All Embeddings ---
modern_embeddings = []
shakespeare_embeddings = []
modern_sentences = []
shakespeare_sentences = []

print(f"Generating embeddings for {SAMPLE_SIZE} pairs...")
for pair in tqdm(data_sample):
    # Get modern embedding
    modern_vec = get_embedding(
        pair['modern_sentence'], model, tokenizer, DEVICE
    )
    modern_embeddings.append(modern_vec)
    modern_sentences.append(pair['modern_sentence']) # Store for hover text

    # Get Shakespearean embedding
    shakespeare_vec = get_embedding(
        pair['shakespeare_sentence'], model, tokenizer, DEVICE
    )
    shakespeare_embeddings.append(shakespeare_vec)
    shakespeare_sentences.append(pair['shakespeare_sentence']) # Store for hover text

# Convert lists to numpy arrays
modern_embeddings = np.array(modern_embeddings)
shakespeare_embeddings = np.array(shakespeare_embeddings)

print("Embeddings generated.")

# --- 5. Run PCA ---
print("Running PCA to reduce to 3 dimensions...")
# Combine all embeddings to fit PCA on the joint distribution
all_embeddings = np.concatenate([modern_embeddings, shakespeare_embeddings])

pca = PCA(n_components=3)
transformed_embeddings = pca.fit_transform(all_embeddings)

# Split back into modern and shakespearean
modern_3d = transformed_embeddings[:SAMPLE_SIZE]
shakespeare_3d = transformed_embeddings[SAMPLE_SIZE:]

print("PCA complete.")

# --- 6. Visualize with Plotly ---
print("Generating 3D plot with arrows... Check your browser.")

# Trace 1: Modern (Blue)
trace_modern = go.Scatter3d(
    x=modern_3d[:, 0],
    y=modern_3d[:, 1],
    z=modern_3d[:, 2],
    mode='markers',
    name='Modern Sentences', # Updated name for legend
    marker=dict(color='blue', size=4, opacity=0.7),
    text=[f"Modern: {s}" for s in modern_sentences], # Hover text
    hoverinfo='text'
)

# Trace 3: Style Vectors (Green Dotted Lines)
lines_x, lines_y, lines_z = [], [], []
for i in range(SAMPLE_SIZE):
    # Add start point (modern)
    lines_x.append(modern_3d[i, 0])
    lines_y.append(modern_3d[i, 1])
    lines_z.append(modern_3d[i, 2])

    # Add end point (shakespearean)
    lines_x.append(shakespeare_3d[i, 0])
    lines_y.append(shakespeare_3d[i, 1])
    lines_z.append(shakespeare_3d[i, 2])

    # Add None to break the line
    lines_x.append(None)
    lines_y.append(None)
    lines_z.append(None)

trace_vectors_lines = go.Scatter3d(
    x=lines_x,
    y=lines_y,
    z=lines_z,
    mode='lines',
    name='Style Vectors', # Name for legend
    line=dict(color='green', width=2, dash='dot'), # <-- CHANGE: Color set to 'green'
    opacity=0.6
)

# Trace 4: Arrowheads (Cones)
arrow_cones = []
for i in range(SAMPLE_SIZE):
    start_point = modern_3d[i]
    end_point = shakespeare_3d[i]

    # Calculate direction vector
    direction = end_point - start_point

    # Calculate the overall scale of the PCA space
    # This helps in setting an appropriate arrow size
    scale_factor = np.mean(np.abs(transformed_embeddings.max(axis=0) - transformed_embeddings.min(axis=0)))
    arrow_length = ARROW_SIZE_SCALE * scale_factor # Length of the cone

    # Normalize direction for 'u', 'v', 'w'
    # Add a small epsilon to avoid division by zero if start == end
    norm_direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Cone position (end_point - a bit back along the direction to ensure it's on the line)
    cone_base_pos = end_point - norm_direction * (arrow_length / 2) # Adjust cone position to be at the end

    arrow_cones.append(go.Cone(
        x=[cone_base_pos[0]],
        y=[cone_base_pos[1]],
        z=[cone_base_pos[2]],
        u=[norm_direction[0]],
        v=[norm_direction[1]],
        w=[norm_direction[2]],
        sizemode="absolute",
        sizeref=arrow_length, # Use calculated length
        showscale=False,
        colorscale=[[0, 'green'], [1, 'green']], # <-- CHANGE: Color set to 'green'
        opacity=0.8,
        # Name this separately for the legend only if you want individual cone entries
        # For a cleaner legend, we'll combine it with the 'Style Vectors' entry
        hoverinfo='skip'
    ))

# Create layout
layout = go.Layout(
    title='Modern vs. Shakespearean Sentence Embeddings (PCA) with Style Vectors',
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    hovermode='closest', # Ensures good hover interaction
    showlegend=True # Ensure legend is shown
)

# Generate figure and show
fig = go.Figure(data=[trace_modern, trace_shakespeare, trace_vectors_lines] + arrow_cones, layout=layout)
fig.show()
