"""
Contrastive Loss Functions for Equation-CLIP

Implements CLIP-style contrastive losses including:
- Symmetric InfoNCE (CLIP loss)
- Hard negative mining variants
- Temperature-scaled losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPLoss(nn.Module):
    """
    CLIP-style symmetric contrastive loss (InfoNCE).

    Given a batch of (equation, text) pairs, maximize similarity
    of positive pairs and minimize similarity of negative pairs.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        temperature_init: float = 0.07,
        temperature_max: float = 100.0
    ):
        """
        Initialize CLIP loss.

        Args:
            temperature: Initial temperature for scaling similarities
            learnable_temperature: Whether temperature is learnable
            temperature_init: Initial value for learnable temperature
            temperature_max: Maximum value for temperature (clipping)
        """
        super().__init__()

        if learnable_temperature:
            # Learnable temperature (as in CLIP paper)
            self.temperature = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1.0 / temperature_init))
            )
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        self.temperature_max = temperature_max
        self.learnable = learnable_temperature

        logger.info(f"Initialized CLIPLoss with temperature={temperature}, "
                   f"learnable={learnable_temperature}")

    def get_temperature(self) -> float:
        """Get current temperature value."""
        if self.learnable:
            # Temperature is stored as log, exponentiate and clip
            temp = torch.exp(self.temperature).clamp(max=self.temperature_max)
            return temp.item()
        else:
            return self.temperature.item()

    def forward(
        self,
        equation_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        return_similarity_matrix: bool = False
    ) -> torch.Tensor:
        """
        Compute CLIP loss.

        Args:
            equation_embeds: Normalized equation embeddings (batch_size, dim)
            text_embeds: Normalized text embeddings (batch_size, dim)
            return_similarity_matrix: Whether to return similarity matrix

        Returns:
            Loss value (scalar) or (loss, similarity_matrix) if return_similarity_matrix=True
        """
        batch_size = equation_embeds.shape[0]

        # Normalize embeddings (L2 normalization)
        equation_embeds = F.normalize(equation_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Compute temperature
        if self.learnable:
            temp = torch.exp(self.temperature).clamp(max=self.temperature_max)
        else:
            temp = self.temperature

        # Compute similarity matrix
        # (batch, dim) @ (dim, batch) -> (batch, batch)
        logits = (equation_embeds @ text_embeds.T) / temp

        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=equation_embeds.device)

        # Symmetric loss
        loss_eq2text = F.cross_entropy(logits, labels)
        loss_text2eq = F.cross_entropy(logits.T, labels)

        loss = (loss_eq2text + loss_text2eq) / 2

        if return_similarity_matrix:
            return loss, logits
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss (equivalent to CLIP but more explicit formulation).

    L = -log[ exp(sim(x,y+)/τ) / Σ_i exp(sim(x,yi)/τ) ]
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature for scaling
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"Initialized InfoNCELoss with temperature={temperature}")

    def forward(
        self,
        query_embeds: torch.Tensor,
        key_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            query_embeds: Query embeddings (batch_size, dim)
            key_embeds: Key embeddings (batch_size, dim)

        Returns:
            Loss value (scalar)
        """
        # Normalize
        query_embeds = F.normalize(query_embeds, p=2, dim=-1)
        key_embeds = F.normalize(key_embeds, p=2, dim=-1)

        # Compute similarities
        logits = (query_embeds @ key_embeds.T) / self.temperature

        # Labels (diagonal)
        labels = torch.arange(len(query_embeds), device=query_embeds.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class HardNegativeCLIPLoss(nn.Module):
    """
    CLIP loss with hard negative mining.

    Focuses on hardest negatives within each batch to improve
    fine-grained discrimination.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 0.5,
        num_hard_negatives: int = 5
    ):
        """
        Initialize hard negative CLIP loss.

        Args:
            temperature: Temperature for scaling
            hard_negative_weight: Weight for hard negatives (0-1)
            num_hard_negatives: Number of hard negatives to mine per sample
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.num_hard_negatives = num_hard_negatives

        self.base_loss = CLIPLoss(temperature=temperature, learnable_temperature=False)

        logger.info(f"Initialized HardNegativeCLIPLoss with "
                   f"hard_weight={hard_negative_weight}")

    def forward(
        self,
        equation_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hard negative CLIP loss.

        Args:
            equation_embeds: Equation embeddings (batch_size, dim)
            text_embeds: Text embeddings (batch_size, dim)

        Returns:
            Loss value (scalar)
        """
        batch_size = equation_embeds.shape[0]

        # Compute base CLIP loss and similarity matrix
        base_loss, logits = self.base_loss(
            equation_embeds,
            text_embeds,
            return_similarity_matrix=True
        )

        # Mine hard negatives
        # For each sample, find hardest negatives (highest similarity among negatives)
        identity_mask = torch.eye(batch_size, device=logits.device).bool()

        # Mask out positive pairs
        negative_logits = logits.clone()
        negative_logits[identity_mask] = -float('inf')

        # Get top-k hardest negatives
        hard_negatives, _ = torch.topk(
            negative_logits,
            k=min(self.num_hard_negatives, batch_size - 1),
            dim=-1
        )

        # Compute loss on hard negatives
        labels = torch.zeros(batch_size, device=logits.device, dtype=torch.long)
        positive_logits = logits[identity_mask].unsqueeze(-1)

        # Combine positive and hard negative logits
        hard_negative_logits = torch.cat([positive_logits, hard_negatives], dim=-1)
        hard_loss = F.cross_entropy(hard_negative_logits, labels)

        # Combine base loss and hard negative loss
        total_loss = (
            (1 - self.hard_negative_weight) * base_loss +
            self.hard_negative_weight * hard_loss
        )

        return total_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for equation-text pairs.

    Alternative to contrastive learning, uses (anchor, positive, negative) triplets.
    """

    def __init__(self, margin: float = 0.2):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
        logger.info(f"Initialized TripletLoss with margin={margin}")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings (batch_size, dim)
            positive: Positive embeddings (batch_size, dim)
            negative: Negative embeddings (batch_size, dim)

        Returns:
            Loss value (scalar)
        """
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        # Compute distances
        pos_dist = (anchor - positive).pow(2).sum(dim=-1)
        neg_dist = (anchor - negative).pow(2).sum(dim=-1)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        return loss


def compute_retrieval_metrics(
    equation_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: list = [1, 5, 10]
) -> dict:
    """
    Compute retrieval metrics (Recall@K) for evaluation.

    Args:
        equation_embeds: Equation embeddings (N, dim)
        text_embeds: Text embeddings (N, dim)
        k_values: List of K values for Recall@K

    Returns:
        Dictionary of metrics
    """
    # Normalize
    equation_embeds = F.normalize(equation_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    # Compute similarity matrix
    similarity = equation_embeds @ text_embeds.T  # (N, N)

    metrics = {}

    # Text-to-Equation retrieval
    for k in k_values:
        _, top_k_indices = torch.topk(similarity, k=k, dim=0)
        correct = torch.any(
            top_k_indices == torch.arange(len(text_embeds), device=similarity.device).unsqueeze(0),
            dim=0
        )
        recall = correct.float().mean().item()
        metrics[f'text2eq_recall@{k}'] = recall

    # Equation-to-Text retrieval
    for k in k_values:
        _, top_k_indices = torch.topk(similarity, k=k, dim=1)
        correct = torch.any(
            top_k_indices == torch.arange(len(equation_embeds), device=similarity.device).unsqueeze(1),
            dim=1
        )
        recall = correct.float().mean().item()
        metrics[f'eq2text_recall@{k}'] = recall

    # Mean Reciprocal Rank (MRR)
    ranks = torch.argsort(torch.argsort(similarity, dim=1, descending=True), dim=1)
    correct_ranks = ranks[torch.arange(len(equation_embeds)), torch.arange(len(equation_embeds))]
    mrr = (1.0 / (correct_ranks.float() + 1)).mean().item()
    metrics['mrr'] = mrr

    return metrics


if __name__ == "__main__":
    # Test loss functions
    logger.info("Testing Loss Functions...")

    batch_size = 8
    dim = 256

    # Create dummy embeddings
    equation_embeds = torch.randn(batch_size, dim)
    text_embeds = torch.randn(batch_size, dim)

    # Test CLIP loss
    clip_loss = CLIPLoss(temperature=0.07, learnable_temperature=True)
    loss = clip_loss(equation_embeds, text_embeds)
    logger.info(f"CLIP Loss: {loss.item():.4f}")
    logger.info(f"Temperature: {clip_loss.get_temperature():.4f}")

    # Test InfoNCE loss
    infonce_loss = InfoNCELoss(temperature=0.07)
    loss = infonce_loss(equation_embeds, text_embeds)
    logger.info(f"InfoNCE Loss: {loss.item():.4f}")

    # Test hard negative loss
    hard_loss = HardNegativeCLIPLoss(temperature=0.07)
    loss = hard_loss(equation_embeds, text_embeds)
    logger.info(f"Hard Negative Loss: {loss.item():.4f}")

    # Test retrieval metrics
    metrics = compute_retrieval_metrics(equation_embeds, text_embeds)
    logger.info(f"Retrieval Metrics: {metrics}")

    logger.info("✓ Loss function tests passed!")
