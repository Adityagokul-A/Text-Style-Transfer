import torch
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def get_embedding(text, model, tokenizer, device):
    """Compute mean-pooled encoder embedding for a sentence."""
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        ).to(device)

        encoder_outputs = model.base_model.encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
        hidden = encoder_outputs.last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1).expand(hidden.shape)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.squeeze().cpu().numpy()

def visualize_embeddings(pairs, model, tokenizer, device, title="Style Transfer Embeddings"):
    """
    Visualize embedding shifts between input and output sentences.
    Accepts a list of (source_sentence, target_sentence) pairs.
    """
    source_embs, target_embs = [], []
    for src, tgt in pairs:
        source_embs.append(get_embedding(src, model, tokenizer, device))
        target_embs.append(get_embedding(tgt, model, tokenizer, device))

    source_embs = np.array(source_embs)
    target_embs = np.array(target_embs)

    # Combine all embeddings to fit PCA
    all_embs = np.concatenate([source_embs, target_embs])

    # Determine valid number of components dynamically
    num_samples, num_features = all_embs.shape
    n_components = min(3, num_samples, num_features)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(all_embs)

    src_3d = reduced[:len(pairs)]
    tgt_3d = reduced[len(pairs):]

    # --- Build 2D or 3D plot dynamically ---
    if n_components == 2:
        trace_src = go.Scatter(
            x=src_3d[:, 0],
            y=src_3d[:, 1],
            mode="markers",
            name="Source",
            marker=dict(color="blue", size=8)
        )
        trace_tgt = go.Scatter(
            x=tgt_3d[:, 0],
            y=tgt_3d[:, 1],
            mode="markers",
            name="Transformed",
            marker=dict(color="red", size=8)
        )

        lines_x, lines_y = [], []
        for i in range(len(pairs)):
            lines_x += [src_3d[i, 0], tgt_3d[i, 0], None]
            lines_y += [src_3d[i, 1], tgt_3d[i, 1], None]

        trace_lines = go.Scatter(
            x=lines_x,
            y=lines_y,
            mode="lines",
            name="Style Shift",
            line=dict(color="green", width=2, dash="dot")
        )

        fig = go.Figure(data=[trace_src, trace_tgt, trace_lines])
        fig.update_layout(
            title=title + " (2D projection)",
            xaxis_title="PCA 1",
            yaxis_title="PCA 2",
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )

    else:
        trace_src = go.Scatter3d(
            x=src_3d[:, 0], y=src_3d[:, 1], z=src_3d[:, 2],
            mode="markers", name="Source", marker=dict(color="blue", size=4)
        )
        trace_tgt = go.Scatter3d(
            x=tgt_3d[:, 0], y=tgt_3d[:, 1], z=tgt_3d[:, 2],
            mode="markers", name="Transformed", marker=dict(color="red", size=4)
        )

        lines_x, lines_y, lines_z = [], [], []
        for i in range(len(pairs)):
            lines_x += [src_3d[i, 0], tgt_3d[i, 0], None]
            lines_y += [src_3d[i, 1], tgt_3d[i, 1], None]
            lines_z += [src_3d[i, 2], tgt_3d[i, 2], None]

        trace_lines = go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode="lines", name="Style Shift",
            line=dict(color="green", width=2, dash="dot")
        )

        fig = go.Figure(data=[trace_src, trace_tgt, trace_lines])
        fig.update_layout(
            title=title + " (3D projection)",
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )

    return fig