#!/usr/bin/env python3
"""
Evaluation script for Equation-CLIP retrieval performance.
"""

import os
import sys
import json
import logging
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.clip_model import build_equation_clip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EquationCLIPSequenceDataset(Dataset):
    """Dataset for Equation-CLIP using sequence encoder."""
    
    def __init__(self, data_file: str, tokenizer_name: str = 'allenai/scibert_scivocab_uncased', max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        with open(data_file) as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        equation_latex = item['equation']
        description = item['description']
        
        equation_tokens = self.tokenizer(
            equation_latex,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        text_tokens = self.tokenizer(
            description,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'equation_token_ids': equation_tokens['input_ids'].squeeze(0),
            'equation_attention_mask': equation_tokens['attention_mask'].squeeze(0),
            'text_input_ids': text_tokens['input_ids'].squeeze(0),
            'text_attention_mask': text_tokens['attention_mask'].squeeze(0),
            'equation': equation_latex,
            'description': description,
            'id': item.get('id', str(idx))
        }


def compute_retrieval_metrics(similarity_matrix, k_values=[1, 5, 10, 20]):
    """Compute retrieval metrics from similarity matrix."""
    n = similarity_matrix.shape[0]
    metrics = {}
    
    # Equation-to-Text retrieval
    for k in k_values:
        if k > n:
            continue
        _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
        correct = torch.any(
            top_k_indices == torch.arange(n).unsqueeze(1),
            dim=1
        )
        recall = correct.float().mean().item()
        metrics[f'eq2text_recall@{k}'] = recall
    
    # Text-to-Equation retrieval
    for k in k_values:
        if k > n:
            continue
        _, top_k_indices = torch.topk(similarity_matrix.T, k=k, dim=1)
        correct = torch.any(
            top_k_indices == torch.arange(n).unsqueeze(1),
            dim=1
        )
        recall = correct.float().mean().item()
        metrics[f'text2eq_recall@{k}'] = recall
    
    # Mean Reciprocal Rank (MRR)
    ranks = torch.argsort(torch.argsort(similarity_matrix, dim=1, descending=True), dim=1)
    correct_ranks = ranks[torch.arange(n), torch.arange(n)]
    mrr = (1.0 / (correct_ranks.float() + 1)).mean().item()
    metrics['eq2text_mrr'] = mrr
    
    # Median rank
    median_rank = torch.median(correct_ranks.float()).item()
    metrics['eq2text_median_rank'] = median_rank
    
    return metrics


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    
    all_equation_embeds = []
    all_text_embeds = []
    all_equations = []
    all_descriptions = []
    all_ids = []
    
    logger.info("Encoding all test samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluation'):
            equation_inputs = {
                'token_ids': batch['equation_token_ids'].to(device),
                'attention_mask': batch['equation_attention_mask'].to(device),
                'normalize': True
            }
            
            text_inputs = {
                'input_ids': batch['text_input_ids'].to(device),
                'attention_mask': batch['text_attention_mask'].to(device),
                'normalize': True
            }
            
            outputs = model(
                equation_inputs,
                text_inputs,
                return_loss=False,
                return_embeddings=True
            )
            
            all_equation_embeds.append(outputs['equation_embeds'].cpu())
            all_text_embeds.append(outputs['text_embeds'].cpu())
            all_equations.extend(batch['equation'])
            all_descriptions.extend(batch['description'])
            all_ids.extend(batch['id'])
    
    # Concatenate all embeddings
    equation_embeds = torch.cat(all_equation_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    logger.info(f"Computing similarity matrix for {len(equation_embeds)} samples...")
    similarity = equation_embeds @ text_embeds.T
    
    # Compute metrics
    metrics = compute_retrieval_metrics(similarity)
    
    # Find failure cases (Recall@1 = 0)
    _, top_1_indices = torch.topk(similarity, k=1, dim=1)
    failures = (top_1_indices.squeeze() != torch.arange(len(equation_embeds)))
    failure_indices = torch.where(failures)[0].tolist()
    
    failure_cases = []
    for idx in failure_indices[:20]:  # Get top 20 failure cases
        retrieved_idx = top_1_indices[idx].item()
        failure_cases.append({
            'id': all_ids[idx],
            'true_equation': all_equations[idx],
            'true_description': all_descriptions[idx],
            'retrieved_description': all_descriptions[retrieved_idx],
            'similarity_score': similarity[idx, retrieved_idx].item(),
            'correct_score': similarity[idx, idx].item()
        })
    
    return metrics, failure_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, default='data/dataset/test.json')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=str, default='evaluation_results.json')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model
    model_config = {
        'equation_encoder_type': 'sequence',
        'equation_vocab_size': 30522,
        'equation_hidden_dim': 512,
        'equation_num_layers': 6,
        'text_encoder_type': 'scibert',
        'text_model_name': 'allenai/scibert_scivocab_uncased',
        'text_hidden_dim': 768,
        'freeze_text_layers': 6,
        'embedding_dim': 256,
        'dropout': 0.1,
        'temperature': 0.07,
    }
    
    model = build_equation_clip(model_config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Load test data
    test_dataset = EquationCLIPSequenceDataset(args.test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    logger.info("Starting evaluation...")
    metrics, failure_cases = evaluate(model, test_loader, device)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    for metric, value in sorted(metrics.items()):
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"\nNumber of failure cases (Recall@1=0): {len(failure_cases)}")
    
    # Save results
    results = {
        'metrics': metrics,
        'failure_cases': failure_cases,
        'num_samples': len(test_dataset)
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")
    
    # Print some failure examples
    if failure_cases:
        logger.info("\n" + "="*60)
        logger.info("SAMPLE FAILURE CASES")
        logger.info("="*60)
        for i, case in enumerate(failure_cases[:5], 1):
            logger.info(f"\nFailure Case {i}:")
            logger.info(f"  Equation: {case['true_equation']}")
            logger.info(f"  True Description: {case['true_description']}")
            logger.info(f"  Retrieved Description: {case['retrieved_description']}")
            logger.info(f"  Correct Score: {case['correct_score']:.4f}, Retrieved Score: {case['similarity_score']:.4f}")


if __name__ == '__main__':
    main()
