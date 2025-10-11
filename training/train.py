"""
Training Script for Equation-CLIP

Implements curriculum learning with three phases:
1. Warm-up on textbook data
2. Main training on arXiv data
3. Hard negative mining
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
import wandb
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_model import build_equation_clip
from data.dataset import create_dataloaders
from models.losses import compute_retrieval_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EquationCLIPTrainer:
    """Trainer for Equation-CLIP model."""
    
    def __init__(self, config: dict):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Number of GPUs: {self.num_gpus}")
        
        # Build model
        logger.info("Building Equation-CLIP model...")
        self.model = build_equation_clip(config)
        
        # Multi-GPU setup
        if self.num_gpus > 1:
            logger.info(f"Using DataParallel with {self.num_gpus} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Initialize wandb
        wandb.init(
            project="equation-clip",
            name=f"arxiv_run_{config.get('equation_encoder_type', 'gnn')}",
            config=config
        )
        wandb.watch(self.model, log='all', log_freq=100)

        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 50),
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = config.get('use_amp', True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move to device
            equation_inputs = {
                'node_types': batch['equation_node_types'].to(self.device),
                'node_values': batch['equation_node_values'].to(self.device),
                'edge_index': batch['equation_edge_index'].to(self.device),
                'batch': batch['equation_batch'].to(self.device)
            }
            
            text_inputs = {
                'input_ids': batch['text_input_ids'].to(self.device),
                'attention_mask': batch['text_attention_mask'].to(self.device)
            }
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(equation_inputs, text_inputs, return_loss=True)
                    loss = outputs['loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(equation_inputs, text_inputs, return_loss=True)
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / num_batches})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_eq_embeds = []
        all_text_embeds = []
        
        for batch in tqdm(val_loader, desc="Validating"):
            # Move to device
            equation_inputs = {
                'node_types': batch['equation_node_types'].to(self.device),
                'node_values': batch['equation_node_values'].to(self.device),
                'edge_index': batch['equation_edge_index'].to(self.device),
                'batch': batch['equation_batch'].to(self.device)
            }
            
            text_inputs = {
                'input_ids': batch['text_input_ids'].to(self.device),
                'attention_mask': batch['text_attention_mask'].to(self.device)
            }
            
            # Forward pass
            outputs = self.model(
                equation_inputs,
                text_inputs,
                return_loss=True,
                return_embeddings=True
            )
            
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            # Collect embeddings for metrics
            all_eq_embeds.append(outputs['equation_embeds'].cpu())
            all_text_embeds.append(outputs['text_embeds'].cpu())
        
        avg_loss = total_loss / num_batches
        
        # Compute retrieval metrics
        all_eq_embeds = torch.cat(all_eq_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)
        metrics = compute_retrieval_metrics(all_eq_embeds, all_text_embeds)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting Equation-CLIP Training")
        logger.info("=" * 60)
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            logger.info(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f}")
            logger.info(f"Retrieval Metrics: {metrics}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.config.get('save_every', 5) == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Save training log
            log_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics,
                'lr': self.scheduler.get_last_lr()[0]
            }
            
            log_file = self.log_dir / 'training_log.jsonl'
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)


def main():
    """Main training function."""
    # Configuration
    config = {
        # Model config
        'equation_encoder_type': 'gnn',
        'equation_vocab_size': 5000,
        'equation_hidden_dim': 512,
        'equation_num_layers': 3,
        'text_encoder_type': 'scibert',
        'text_model_name': 'allenai/scibert_scivocab_uncased',
        'text_hidden_dim': 768,
        'freeze_text_layers': 6,
        'embedding_dim': 256,
        'dropout': 0.1,
        'temperature': 0.07,
        'learnable_temperature': True,
        
        # Training config
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'batch_size': 64,  # Per GPU
        'num_epochs': 50,
        'use_amp': True,
        'save_every': 5,
        
        # Data config
        'train_file': './data/dataset/train.json',
        'val_file': './data/dataset/val.json',
        'num_workers': 4,
        
        # Paths
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs'
    }
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_file=config['train_file'],
        val_file=config['val_file'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create trainer
    trainer = EquationCLIPTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader, config['num_epochs'])


if __name__ == "__main__":
    main()
