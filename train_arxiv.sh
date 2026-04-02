#!/bin/bash
cd ~/equation-clip
python3 -m torch.distributed.run --nproc_per_node=8 \
  training/train_sequence_v2.py \
  --data-dir data/dataset \
  --output-dir checkpoints_arxiv \
  --epochs 50 \
  --batch-size 64 \
  --use-wandb
