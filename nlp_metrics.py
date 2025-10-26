#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-26T11:09:54.094Z
"""

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from collections import Counter

def get_bert_embeddings(sentences, model, tokenizer, device):
    # Tokenize and encode
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the last hidden states as embeddings
    # shape: [batch_size, seq_len, hidden_dim]
    hidden_states = outputs.last_hidden_state
    return hidden_states, inputs["attention_mask"]

def bertscore_single(candidate, reference, model, tokenizer, device):
    # Get embeddings
    cand_emb, cand_mask = get_bert_embeddings([candidate], model, tokenizer, device)
    ref_emb, ref_mask = get_bert_embeddings([reference], model, tokenizer, device)
    
    # Squeeze batch dimension
    cand_emb = cand_emb.squeeze(0)[cand_mask.squeeze(0) == 1]
    ref_emb = ref_emb.squeeze(0)[ref_mask.squeeze(0) == 1]
    
    # Get cosine similarity matrix
    similarity = cosine_similarity(cand_emb.cpu().numpy(), ref_emb.cpu().numpy())
    
    # Precision: for each candidate token, max similarity to ref tokens
    precision = np.mean(np.max(similarity, axis=1))
    # Recall: for each reference token, max similarity to cand tokens
    recall = np.mean(np.max(similarity, axis=0))
    # F1
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def hungarian_algorithm(cost_matrix):
    # Basic implementation for square matrices.
    # For non-square, pad with zeros.
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def get_word_embeddings(sentence, tokenizer, model, device):
    # Tokenize and get outputs
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Remove batch dim; take only tokens before [PAD]
    mask = inputs['attention_mask'][0].bool()
    embeddings = outputs.last_hidden_state[0, mask]
    return embeddings.cpu().numpy()

def pairwise_distances(X, Y):
    # X: (n, d), Y: (m, d)
    # Returns (n, m) Euclidean distance matrix
    n, m = X.shape[0], Y.shape[0]
    dists = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dists[i, j] = np.linalg.norm(X[i] - Y[j])
    return dists

def emd_distance(emb_actual, emb_generated):
    # Pad to square matrix for Hungarian if necessary
    n_actual, n_generated = emb_actual.shape[0], emb_generated.shape[0]
    cost_matrix = pairwise_distances(emb_actual, emb_generated)
    if n_actual != n_generated:
        # Pad cost_matrix to square with zeros
        size = max(n_actual, n_generated)
        padded = np.zeros((size, size))
        padded[:n_actual, :n_generated] = cost_matrix
        cost_matrix = padded
    # Optimal assignment for minimal cost
    row_ind, col_ind = hungarian_algorithm(cost_matrix)
    # Take only assignments within original shape
    total_cost = 0
    for i, j in zip(row_ind, col_ind):
        if i < n_actual and j < n_generated:
            total_cost += cost_matrix[i, j]
    # Average cost per assigned word (EMD)
    emd = total_cost / min(n_actual, n_generated)
    return emd


def count_clip(candidate, references, n):
    # Count n-grams in candidate and max in refs
    cand_ngrams = Counter(ngrams(candidate, n))
    max_ref_ngrams = Counter()
    for ref in references:
        ref_ngrams = Counter(ngrams(ref, n))
        for ngram in ref_ngrams:
            max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], ref_ngrams[ngram])
    # Clip candidate ngram counts by max in any reference
    clip_counts = {ng: min(count, max_ref_ngrams[ng]) for ng, count in cand_ngrams.items()}
    return sum(clip_counts.values()), max(1, sum(cand_ngrams.values()))

def brevity_penalty(cand_len, ref_lens):
    # BP as in BLEU paper
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - cand_len), ref_len))
    if cand_len > closest_ref_len:
        return 1
    elif cand_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / cand_len)

def compute_bleu(candidate, references, max_n=4):
    # Tokenize candidate and refs
    if isinstance(candidate, str):
        candidate = candidate.split()
    refs = [ref.split() if isinstance(ref, str) else ref for ref in references]
    # N-gram precisions
    p_n = []
    for n in range(1, max_n + 1):
        match, total = count_clip(candidate, refs, n)
        p_n.append(match / total if total > 0 else 0)
    # Geometric mean of precisions
    if min(p_n) > 0:
        score = math.exp(sum(math.log(p) for p in p_n) / max_n)
    else:
        score = 0
    # Brevity penalty
    bp = brevity_penalty(len(candidate), [len(ref) for ref in refs])
    bleu = bp * score
    return bleu

def rouge_n(candidate, reference, n=1):
    # Tokenize
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)])
    ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])

    # Count overlap
    overlap = sum((cand_ngrams & ref_ngrams).values())
    recall = overlap / max(1, sum(ref_ngrams.values()))
    precision = overlap / max(1, sum(cand_ngrams.values()))
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    return {'recall': recall, 'precision': precision, 'f1': f1}

def lcs_length(x, y):
    # Classic LCS DP algorithm
    m, n = len(x), len(y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def rouge_l(candidate, reference):
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    lcs = lcs_length(cand_tokens, ref_tokens)
    recall = lcs / max(1, len(ref_tokens))
    precision = lcs / max(1, len(cand_tokens))
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    return {'recall': recall, 'precision': precision, 'f1': f1}