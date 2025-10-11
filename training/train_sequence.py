#!/usr/bin/env python3
"""
Training script for Equation-CLIP with sequence encoder (LaTeX tokens).
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.clip_model import build_equation_clip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, training without logging")


class EquationCLIPSequenceDataset(Dataset):
    """Dataset for Equation-CLIP using sequence encoder."""
    
    def __init__(self, data_file: str, tokenizer_name: str = 'allenai/scibert_scivocab_uncased', max_length: int = 128):
        self.data_file = data_file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Load data
        with open(data_file) as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get LaTeX equation and description
        equation_latex = item['equation']
        description = item['description']
        
        # Tokenize equation (treat LaTeX as sequence)
        equation_tokens = self.tokenizer(
            equation_latex,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize description
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
        }


class EquationCLIPTrainer:
    """Trainer for Equation-CLIP."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build model with sequence encoder
        model_config = {
            'equation_encoder_type': 'sequence',
            'equation_vocab_size': 30522,  # SciBERT vocab size
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
        
        self.model = build_equation_clip(model_config)
        
        # Multi-GPU setup with DDP
        self.num_gpus = torch.cuda.device_count()
        self.use_ddp = self.num_gpus > 1 and 'RANK' in os.environ
        
        if self.use_ddp:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
            logger.info(f"Using DDP with {self.world_size} GPUs, local_rank={self.local_rank}")
        else:
            logger.info(f"GPUs available: {self.num_gpus}, using single GPU mode")
            self.model = self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Initialize wandb (only on rank 0)
        is_main_process = not self.use_ddp or self.local_rank == 0
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE and is_main_process
        if self.use_wandb:
            wandb.init(
                project='equation-clip',
                name='arxiv_sequence_encoder_8gpu',
                config=config
            )
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Prepare inputs
            equation_inputs = {
                'token_ids': batch['equation_token_ids'].to(self.device),
                'attention_mask': batch['equation_attention_mask'].to(self.device),
                'normalize': True
            }
            
            text_inputs = {
                'input_ids': batch['text_input_ids'].to(self.device),
                'attention_mask': batch['text_attention_mask'].to(self.device),
                'normalize': True
            }
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(equation_inputs, text_inputs, return_loss=True)
            
            loss = outputs['loss']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Log
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        all_equation_embeds = []
        all_text_embeds = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                equation_inputs = {
                    'token_ids': batch['equation_token_ids'].to(self.device),
                    'attention_mask': batch['equation_attention_mask'].to(self.device),
                    'normalize': True
                }
                
                text_inputs = {
                    'input_ids': batch['text_input_ids'].to(self.device),
                    'attention_mask': batch['text_attention_mask'].to(self.device),
                    'normalize': True
                }
                
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        equation_inputs,
                        text_inputs,
                        return_loss=True,
                        return_embeddings=True
                    )
                
                total_loss += outputs['loss'].item()
                all_equation_embeds.append(outputs['equation_embeds'].cpu())
                all_text_embeds.append(outputs['text_embeds'].cpu())
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute retrieval metrics
        equation_embeds = torch.cat(all_equation_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)
        
        similarity = equation_embeds @ text_embeds.T
        
        # Recall@K
        recalls = {}
        for k in [1, 5, 10]:
            _, top_k_indices = torch.topk(similarity, k=k, dim=1)
            correct = torch.any(
                top_k_indices == torch.arange(len(equation_embeds)).unsqueeze(1),
                dim=1
            )
            recall = correct.float().mean().item()
            recalls[f'recall@{k}'] = recall
        
        logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}")
        logger.info(f"Recall@1: {recalls['recall@1']:.4f}, "
                   f"Recall@5: {recalls['recall@5']:.4f}, "
                   f"Recall@10: {recalls['recall@10']:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                **recalls,
                'epoch': epoch
            })
        
        return avg_loss, recalls
    
    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        all_equation_embeds = []
        all_text_embeds = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                equation_inputs = {
                    'token_ids': batch['equation_token_ids'].to(self.device),
                    'attention_mask': batch['equation_attention_mask'].to(self.device),
                    'normalize': True
                }
                
                text_inputs = {
                    'input_ids': batch['text_input_ids'].to(self.device),
                    'attention_mask': batch['text_attention_mask'].to(self.device),
                    'normalize': True
                }
                
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        equation_inputs,
                        text_inputs,
                        return_loss=True,
                        return_embeddings=True
                    )
                
                total_loss += outputs['loss'].item()
                all_equation_embeds.append(outputs['equation_embeds'].cpu())
                all_text_embeds.append(outputs['text_embeds'].cpu())
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute retrieval metrics
        equation_embeds = torch.cat(all_equation_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)
        
        similarity = equation_embeds @ text_embeds.T
        
        # Recall@K
        recalls = {}
        for k in [1, 5, 10]:
            _, top_k_indices = torch.topk(similarity, k=k, dim=1)
            correct = torch.any(
                top_k_indices == torch.arange(len(equation_embeds)).unsqueeze(1),
                dim=1
            )
            recall = correct.float().mean().item()
            recalls[f'recall@{k}'] = recall
        
        logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}")
        logger.info(f"Recall@1: {recalls['recall@1']:.4f}, "
                   f"Recall@5: {recalls['recall@5']:.4f}, "
                   f"Recall@10: {recalls['recall@10']:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                **recalls,
                'epoch': epoch
            })
        
        return avg_loss, recalls
    
    def train(self, train_loader, val_loader, epochs, save_dir):
        is_main_process = not self.use_ddp or self.local_rank == 0
        """Full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info(f"{'='*60}")
            
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, recalls = self.validate(val_loader, epoch)
            
            self.scheduler.step()
            
            # Save checkpoint
            if is_main_process and (epoch % 5 == 0 or val_loss < best_val_loss):
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
                model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'recalls': recalls
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = save_dir / 'best_model.pt'
                    torch.save(model_to_save.state_dict(), best_path)
                    logger.info(f"New best model saved to {best_path}")
        
        logger.info("\nTraining completed!")
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/dataset')
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--use-wandb', action='store_true')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'use_wandb': args.use_wandb
    }
    
    # Create datasets
    train_dataset = EquationCLIPSequenceDataset(
        os.path.join(args.data_dir, 'train.json')
    )
    val_dataset = EquationCLIPSequenceDataset(
        os.path.join(args.data_dir, 'val.json')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"GPUs available: {torch.cuda.device_count()}")
    
    # Train
    trainer = EquationCLIPTrainer(config)
    trainer.train(train_loader, val_loader, args.epochs, args.output_dir)


if __name__ == '__main__':
    main()
