
import json
import torch
import numpy as np
import random
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from tqdm import tqdm
from visual_embeddings_utils import get_embedding

def visualize(args):
    # --- 1. Configuration ---
    RANDOM_SEED = 42
    ARROW_SIZE_SCALE = 0.05

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- 2. Model & Data Loading ---
    print("Loading model and tokenizer...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    model.to(DEVICE)
    model.eval()

    print("Loading and sampling dataset...")
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if len(data) < args.sample_size:
                print(f"Warning: Sample size ({args.sample_size}) is larger than dataset ({len(data)}). Using all data.")
                data_sample = data
                args.sample_size = len(data)
            else:
                data_sample = random.sample(data, args.sample_size)
        print(f"Loaded {len(data)} pairs, using {args.sample_size} for visualization.")
    except FileNotFoundError:
        print(f"ERROR: Could not find {args.dataset_path}.")
        exit()

    # --- 3. Generate All Embeddings ---
    source_embeddings = []
    target_embeddings = []
    source_sentences = []
    target_sentences = []
    
    source_lang, target_lang = args.style.split('_to_')

    print(f"Generating embeddings for {args.sample_size} pairs...")
    for pair in tqdm(data_sample):
        source_text = pair[f'{source_lang}_sentence']
        target_text = pair[f'{target_lang}_sentence']

        source_vec = get_embedding(source_text, model, tokenizer, DEVICE)
        source_embeddings.append(source_vec)
        source_sentences.append(source_text)

        target_vec = get_embedding(target_text, model, tokenizer, DEVICE)
        target_embeddings.append(target_vec)
        target_sentences.append(target_text)

    source_embeddings = np.array(source_embeddings)
    target_embeddings = np.array(target_embeddings)

    print("Embeddings generated.")

    # --- 4. Run PCA ---
    print("Running PCA to reduce to 3 dimensions...")
    all_embeddings = np.concatenate([source_embeddings, target_embeddings])

    pca = PCA(n_components=3)
    transformed_embeddings = pca.fit_transform(all_embeddings)

    source_3d = transformed_embeddings[:args.sample_size]
    target_3d = transformed_embeddings[args.sample_size:]

    print("PCA complete.")

    # --- 5. Visualize with Plotly ---
    print("Generating 3D plot with arrows... Check your browser.")

    trace_source = go.Scatter3d(
        x=source_3d[:, 0],
        y=source_3d[:, 1],
        z=source_3d[:, 2],
        mode='markers',
        name=f'{source_lang.capitalize()} Sentences',
        marker=dict(color='blue', size=4, opacity=0.7),
        text=[f"{source_lang.capitalize()}: {s}" for s in source_sentences],
        hoverinfo='text'
    )

    trace_target = go.Scatter3d(
        x=target_3d[:, 0],
        y=target_3d[:, 1],
        z=target_3d[:, 2],
        mode='markers',
        name=f'{target_lang.capitalize()} Sentences',
        marker=dict(color='red', size=4, opacity=0.7),
        text=[f"{target_lang.capitalize()}: {s}" for s in target_sentences],
        hoverinfo='text'
    )

    lines_x, lines_y, lines_z = [], [], []
    for i in range(args.sample_size):
        lines_x.extend([source_3d[i, 0], target_3d[i, 0], None])
        lines_y.extend([source_3d[i, 1], target_3d[i, 1], None])
        lines_z.extend([source_3d[i, 2], target_3d[i, 2], None])

    trace_vectors_lines = go.Scatter3d(
        x=lines_x,
        y=lines_y,
        z=lines_z,
        mode='lines',
        name='Style Vectors',
        line=dict(color='green', width=2, dash='dot'),
        opacity=0.6
    )

    arrow_cones = []
    for i in range(args.sample_size):
        start_point = source_3d[i]
        end_point = target_3d[i]
        direction = end_point - start_point
        scale_factor = np.mean(np.abs(transformed_embeddings.max(axis=0) - transformed_embeddings.min(axis=0)))
        arrow_length = ARROW_SIZE_SCALE * scale_factor
        norm_direction = direction / (np.linalg.norm(direction) + 1e-8)
        cone_base_pos = end_point - norm_direction * (arrow_length / 2)

        arrow_cones.append(go.Cone(
            x=[cone_base_pos[0]], y=[cone_base_pos[1]], z=[cone_base_pos[2]],
            u=[norm_direction[0]], v=[norm_direction[1]], w=[norm_direction[2]],
            sizemode="absolute", sizeref=arrow_length, showscale=False,
            colorscale=[[0, 'green'], [1, 'green']], opacity=0.8, hoverinfo='skip'
        ))

    layout = go.Layout(
        title=f'{source_lang.capitalize()} vs. {target_lang.capitalize()} Sentence Embeddings (PCA)',
        scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'),
        margin=dict(l=0, r=0, b=0, t=40),
        hovermode='closest', showlegend=True
    )

    fig = go.Figure(data=[trace_source, trace_target, trace_vectors_lines] + arrow_cones, layout=layout)
    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize the embedding space of a style transfer model.")
    parser.add_argument("--style", type=str, required=True, help="Style for the transfer (e.g., 'modern_to_shakespeare').")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the LoRA adapter.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset (JSON file).")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of sentence pairs to visualize.")

    args = parser.parse_args()
    visualize(args)

if __name__ == "__main__":
    main()
