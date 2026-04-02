# Quick Start Guide for Equation-CLIP Training

## Prerequisites
✓ GPU instance connected (8x H100 80GB)
✓ Environment set up with PyTorch and dependencies
✓ Repository cloned from GitHub

## Step 1: Prepare Your Data

Before training, you need to create the dataset:

```bash
cd ~/equation-clip

# Step 1: Download arXiv papers
python data/download_arxiv.py

# Step 2: Extract equations from LaTeX
python data/extract_equations.py

# Step 3: Parse equations to operator trees
python data/parse_trees.py

# Step 4: Build train/val/test splits
python data/build_dataset.py
```

This will create:
- `data/dataset/train.json`
- `data/dataset/val.json`
- `data/dataset/test.json`

## Step 2: Start Training

Once you have the dataset:

```bash
cd ~/equation-clip
python training/train.py
```

### Training Configuration

The default config in `training/train.py`:
- **Model**: GNN equation encoder + SciBERT text encoder
- **Batch size**: 64 per GPU (512 total with 8 GPUs)
- **Learning rate**: 3e-4 with cosine decay
- **Epochs**: 50
- **Mixed precision**: Enabled (FP16)

### Monitor Training

Training logs are saved to:
- `logs/training_log.jsonl` - Metrics per epoch
- `checkpoints/checkpoint_epoch_*.pt` - Model checkpoints
- `checkpoints/best_model.pt` - Best model

View training progress:
```bash
tail -f logs/training_log.jsonl
```

## Step 3: Evaluate Model

After training, evaluate on test set:

```bash
python evaluation/evaluate.py --checkpoint checkpoints/best_model.pt
```

## Performance Optimization

### Multi-GPU Training (8x H100)

The training script automatically uses `DataParallel` for multi-GPU.

For even better performance, you can modify to use `DistributedDataParallel`:

```bash
# Use all 8 GPUs with DDP
torchrun --nproc_per_node=8 training/train.py
```

### Batch Size Tuning

With 8x H100 (80GB each), you can use large batch sizes:
- Current: 64 per GPU = 512 total
- Recommended: 128 per GPU = 1024 total

Edit `training/train.py`:
```python
config = {
    'batch_size': 128,  # Increase for better gradients
    ...
}
```

### Expected Training Time

For 200K equation-text pairs:
- **8x H100**: ~2-3 days
- **Single H100**: ~16-24 days

## Troubleshooting

### Out of Memory
Reduce batch size:
```python
config['batch_size'] = 32  # Or 16
```

### Slow Data Loading
Increase workers:
```python
config['num_workers'] = 8  # More parallel loading
```

### Model Not Converging
Try these:
1. Lower learning rate: `config['learning_rate'] = 1e-4`
2. Warm up longer: Add warmup steps
3. Check data quality: Ensure equations parse correctly

## What's Next?

Once training is complete:

1. **Evaluate**: Run full evaluation on test set
2. **Analyze**: Check which physics domains perform best
3. **Deploy**: Use trained model for equation retrieval
4. **Iterate**: Try different architectures (GAT vs GCN, etc.)

## Useful Commands

```bash
# Check GPU usage
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check disk space
df -h

# View latest checkpoint
ls -lh checkpoints/

# Kill training if needed
pkill -f train.py
```

## Research Targets (from your plan)

- **Retrieval Recall@5**: >70% (baseline: TangentCFT ~50%)
- **Zero-shot Classification**: >75%
- **Semantic Similarity**: Spearman ρ > 0.7

Good luck with your research! 🚀
