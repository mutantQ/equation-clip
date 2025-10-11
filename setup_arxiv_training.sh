#!/bin/bash
cd ~/equation-clip/data/dataset
# Backup original files
if [ ! -f train.json.synthetic ]; then
  mv train.json train.json.synthetic
  mv val.json val.json.synthetic  
  mv test.json test.json.synthetic
fi

# Link arxiv files
ln -sf train_arxiv.json train.json
ln -sf val_arxiv.json val.json
ln -sf test_arxiv.json test.json

echo "Switched to arXiv dataset"
ls -lh train.json val.json test.json
