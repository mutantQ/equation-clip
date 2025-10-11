"""
Retrieval Evaluation for Equation-CLIP

Implements Recall@K, MRR, and NDCG metrics.
"""

import torch
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_recall_at_k(similarity_matrix: torch.Tensor, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute Recall@K for retrieval.
    
    Args:
        similarity_matrix: (N, N) similarity matrix where (i,j) = sim(query_i, doc_j)
        k_values: List of K values to compute
    
    Returns:
        Dictionary with recall@k scores
    """
    n = similarity_matrix.size(0)
    recalls = {}
    
    for k in k_values:
        # Get top-k indices for each query
        _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
        
        # Check if correct document (diagonal) is in top-k
        correct_in_top_k = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            correct_in_top_k[i] = i in top_k_indices[i]
        
        recall = correct_in_top_k.float().mean().item()
        recalls[f'recall@{k}'] = recall
    
    return recalls


def compute_mrr(similarity_matrix: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        similarity_matrix: (N, N) similarity matrix
    
    Returns:
        MRR score
    """
    n = similarity_matrix.size(0)
    
    # Get ranks of correct documents
    ranks = []
    for i in range(n):
        # Sort similarities in descending order
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
        
        # Find rank of correct document (index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(1.0 / rank)
    
    mrr = np.mean(ranks)
    return mrr


def compute_ndcg_at_k(similarity_matrix: torch.Tensor, k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ K.
    
    Args:
        similarity_matrix: (N, N) similarity matrix
        k: Cutoff rank
    
    Returns:
        NDCG@K score
    """
    n = similarity_matrix.size(0)
    ndcgs = []
    
    for i in range(n):
        # Get top-k predictions
        _, top_k_indices = torch.topk(similarity_matrix[i], k=k)
        
        # Compute DCG
        dcg = 0.0
        for j, idx in enumerate(top_k_indices):
            if idx == i:  # Correct document
                dcg += 1.0 / np.log2(j + 2)  # j+2 because rank starts at 1
        
        # IDCG (ideal: correct doc at rank 1)
        idcg = 1.0 / np.log2(2)
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs)


def evaluate_retrieval(
    equation_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Comprehensive retrieval evaluation.
    
    Args:
        equation_embeds: Equation embeddings (N, dim)
        text_embeds: Text embeddings (N, dim)
        k_values: K values for recall
    
    Returns:
        Dictionary with all metrics
    """
    # Normalize embeddings
    equation_embeds = torch.nn.functional.normalize(equation_embeds, p=2, dim=-1)
    text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
    
    # Compute similarity matrix
    similarity = equation_embeds @ text_embeds.T
    
    # Compute metrics
    metrics = {}
    
    # Text-to-Equation retrieval
    metrics.update({f't2e_{k}': v for k, v in compute_recall_at_k(similarity.T, k_values).items()})
    metrics['t2e_mrr'] = compute_mrr(similarity.T)
    metrics['t2e_ndcg@10'] = compute_ndcg_at_k(similarity.T, k=10)
    
    # Equation-to-Text retrieval
    metrics.update({f'e2t_{k}': v for k, v in compute_recall_at_k(similarity, k_values).items()})
    metrics['e2t_mrr'] = compute_mrr(similarity)
    metrics['e2t_ndcg@10'] = compute_ndcg_at_k(similarity, k=10)
    
    # Average metrics
    for k in k_values:
        metrics[f'avg_recall@{k}'] = (metrics[f't2e_recall@{k}'] + metrics[f'e2t_recall@{k}']) / 2
    metrics['avg_mrr'] = (metrics['t2e_mrr'] + metrics['e2t_mrr']) / 2
    metrics['avg_ndcg@10'] = (metrics['t2e_ndcg@10'] + metrics['e2t_ndcg@10']) / 2
    
    return metrics


if __name__ == "__main__":
    # Test retrieval metrics
    logger.info("Testing retrieval evaluation...")
    
    # Create dummy embeddings
    n = 100
    dim = 256
    equation_embeds = torch.randn(n, dim)
    text_embeds = torch.randn(n, dim)
    
    # Evaluate
    metrics = evaluate_retrieval(equation_embeds, text_embeds)
    
    logger.info("Retrieval Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
