"""Semantic objectives used by the Work1 image-gradient loss."""

import math

import torch
import torch.nn.functional as F


def sinusoidal_positional_encoding(seq_len, dim, *, device, dtype):
    """Create a positional encoding on the same device as the model logits."""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(seq_len, dim, dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding.to(dtype=dtype).unsqueeze(0)


def semantic_similarity_loss(
    logits,
    labels,
    embedding_matrix,
    weights=None,
    mode="token",
    ignore_index=-100,
    verbose=False,
):
    """Compare predicted expected embeddings with supervised target embeddings."""
    del weights
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    _, sequence_length, _ = logits.shape
    embedding_dim = embedding_matrix.size(1)

    probabilities = torch.softmax(logits, dim=-1)
    expected_embeddings = probabilities @ embedding_matrix
    target_embeddings = embedding_matrix[labels.clamp(min=0)]
    valid_mask = labels.ne(ignore_index)
    valid_count = valid_mask.sum()
    if valid_count == 0:
        return logits.sum() * 0.0

    if mode == "token":
        similarity = F.cosine_similarity(
            expected_embeddings, target_embeddings, dim=-1
        )
    elif mode == "attention":
        expected_embeddings = expected_embeddings + (
            torch.randn_like(expected_embeddings) * 1e-4
        )
        positional = sinusoidal_positional_encoding(
            sequence_length,
            embedding_dim,
            device=expected_embeddings.device,
            dtype=expected_embeddings.dtype,
        )
        predicted = F.normalize(expected_embeddings + 0.01 * positional, dim=-1)
        targets = F.normalize(target_embeddings + 0.01 * positional, dim=-1)
        attention_scores = torch.bmm(predicted, targets.transpose(1, 2)) / 0.5

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=labels.device,
            )
        ).unsqueeze(0)
        attention_mask = causal_mask & valid_mask.unsqueeze(1)
        row_has_valid = attention_mask.any(dim=-1)
        attention_scores = attention_scores.masked_fill(
            ~attention_mask, float("-inf")
        )
        attention_scores = attention_scores.masked_fill(
            ~row_has_valid.unsqueeze(-1), 0.0
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_targets = torch.bmm(attention_weights, target_embeddings)
        similarity = F.cosine_similarity(
            expected_embeddings, attended_targets, dim=-1
        )
        if verbose:
            print("Attention matrix")
            print(attention_weights[0].detach().float())
            print("Similarity")
            print(similarity[0].detach().float())
    else:
        raise ValueError("mode must be 'token' or 'attention'")

    valid_weights = valid_mask.to(similarity.dtype)
    return ((1.0 - similarity) * valid_weights).sum() / valid_count
