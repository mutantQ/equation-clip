# Equation-CLIP Research: Executive Summary

## Key Findings

### 1. Current State-of-the-Art

**Graph Contrastive Learning (GCL)** is now the top-performing approach for mathematical retrieval:
- **Paper**: Wang & Chen (2024) - "The Effectiveness of Graph Contrastive Learning on Mathematical Information Retrieval"
- **Performance**: Consistently exceeds TangentCFT (previous SOTA) on NTCIR-12 benchmark
- **Code**: https://github.com/WangPeiSyuan/GCL-Formula-Retrieval

**TangentCFT** (2019) was the previous state-of-the-art:
- Dual representations: Symbol Layout Trees (SLT) + Operator Trees (OPT)
- FastText embeddings on tuple encodings
- ~50% Recall@5 on NTCIR-12
- **Code**: https://github.com/BehroozMansouri/TangentCFT

### 2. Best Pre-trained Models for Equation-CLIP

**Text Encoder (Recommended): PhysBERT**
- **Paper**: arXiv:2408.09574 (August 2024)
- First physics-specific text embedding model
- Trained on 1.2M arXiv physics papers
- Outperforms SciBERT on physics tasks
- **Status**: Check paper for Hugging Face release

**Text Encoder (Fallback): SciBERT**
- **Paper**: arXiv:1903.10676 (2019)
- Trained on 1.14M scientific papers (3.1B tokens)
- Well-established baseline
- **Hugging Face**: `allenai/scibert_scivocab_uncased`

**Other Options**:
- **MathBERT** (Peng et al., 2021): Formula-specific, trained on math content
- **MathBERTa** (Hugging Face): RoBERTa with LaTeX tokenization

### 3. Equation Representation Methods

**Recommended: Graph Neural Networks (GNNs)**
- Recent work (2024) shows GNNs outperform other approaches
- Flexible: can represent both SLT and OPT as graphs
- Methods: InfoGraph, GraphCL, BGRL

**Alternative: Tree Transformers**
- Linearize tree with positional encodings
- Standard Transformer architecture
- Easier to implement, but potentially less effective

**Hybrid Approach**:
- Combine structural (GNN) and sequential (Transformer) streams
- Fusion layer combines both representations
- Best of both worlds

### 4. Major Benchmarks and Datasets

**Benchmarks**:
- **NTCIR-12 MathIR**: 319,689 Wikipedia articles, 592,443 formulas
- **ARQMath** (CLEF 2020-2022): Math Stack Exchange Q&A retrieval
- **MIRB** (2025): New unified benchmark with 12 datasets

**Datasets for Training**:
- **MathBridge** (2024): 23M formula-description pairs from arXiv
- **arXiv Physics**: Millions of papers with LaTeX source (use LaTeXML for extraction)
- **OpenWebMath**: High-quality math content from web

**Extraction Tools**:
- **LaTeXML**: Converts LaTeX to MathML (1.78M arXiv docs processed)
- **HopTeX**: Batch processing for entire arXiv corpus

### 5. Tree Representations

**Three main types**:

1. **Symbol Layout Tree (SLT)**:
   - Visual/spatial arrangement
   - Captures presentation (LaTeX)
   - Nodes = symbols, Edges = spatial relations

2. **Operator Tree (OPT)**:
   - Semantic/mathematical content
   - Captures meaning (Content MathML)
   - Internal nodes = operators, Leaves = operands

3. **Abstract Syntax Tree (AST)**:
   - General programming language concept
   - Hierarchical operator-operand structure
   - Parentheses implicit in structure

### 6. Evaluation Metrics

**Retrieval**:
- **Recall@K**: Target > 70% at K=5
- **MRR** (Mean Reciprocal Rank)
- **NDCG** (Normalized Discounted Cumulative Gain)

**Semantic Similarity**:
- **Spearman ρ**: Target > 0.7
- Requires human-annotated equation pairs

**Classification**:
- **Accuracy**: Target > 75% for zero-shot domain classification
- **F1 Score**

**Clustering**:
- **ARI** (Adjusted Rand Index): Target > 0.6
- **NMI** (Normalized Mutual Information)
- **Silhouette Score**: Target > 0.5

---

## Recommended Architecture for Equation-CLIP

```
┌─────────────────────────────────────────────────────────┐
│                    Equation-CLIP                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐              ┌─────────────────┐    │
│  │  LaTeX       │              │  Text           │    │
│  │  Equation    │              │  Description    │    │
│  └──────┬───────┘              └────────┬────────┘    │
│         │                               │             │
│         ▼                               ▼             │
│  ┌──────────────┐              ┌─────────────────┐    │
│  │ Parse to OPT │              │   PhysBERT      │    │
│  │  (LaTeXML)   │              │  (pre-trained)  │    │
│  └──────┬───────┘              └────────┬────────┘    │
│         │                               │             │
│         ▼                               ▼             │
│  ┌──────────────┐              ┌─────────────────┐    │
│  │     GNN      │              │  Fine-tune      │    │
│  │   Encoder    │              │   Last Layers   │    │
│  └──────┬───────┘              └────────┬────────┘    │
│         │                               │             │
│         ▼                               ▼             │
│  ┌──────────────┐              ┌─────────────────┐    │
│  │  2-Layer MLP │              │  2-Layer MLP    │    │
│  │  Projection  │              │  Projection     │    │
│  └──────┬───────┘              └────────┬────────┘    │
│         │                               │             │
│         ▼                               ▼             │
│  ┌──────────────┐              ┌─────────────────┐    │
│  │  Equation    │              │  Text           │    │
│  │  Embedding   │◄─────────────┤  Embedding      │    │
│  │  (256-d)     │  InfoNCE     │  (256-d)        │    │
│  └──────────────┘   Loss       └─────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Design Choices

**Equation Encoder**: Graph Neural Network
- Input: Operator Tree (OPT) from LaTeX
- Architecture: GCN or GAT (3-5 layers)
- Pooling: Global mean + max pooling
- Output: 256-d embedding

**Text Encoder**: PhysBERT
- Input: Natural language description
- Fine-tuning: Last 3-6 layers
- Output: 256-d embedding

**Loss Function**: InfoNCE (CLIP-style contrastive loss)
- Temperature: τ = 0.07
- Symmetric loss (equation→text + text→equation)

**Training Strategy**:
- Curriculum learning: Textbooks → arXiv → Hard negatives
- Batch size: 512-1024
- Learning rate: 3e-4 (AdamW)
- Augmentation: Variable renaming, notation variants, paraphrasing

---

## Implementation Roadmap

### Phase 1: Data (Months 1-2)
- Deploy LaTeXML pipeline on arXiv
- Extract 500K equation-description pairs
- Manual annotation of 2K test pairs
- **Deliverable**: Training/validation/test datasets

### Phase 2: Model (Months 3-4)
- Implement GNN equation encoder
- Load PhysBERT, add projection heads
- InfoNCE loss + augmentation pipeline
- **Deliverable**: Working Equation-CLIP

### Phase 3: Training (Month 5)
- Curriculum training + hard negative mining
- Hyperparameter tuning
- Ablation studies
- **Deliverable**: Trained model

### Phase 4: Evaluation (Month 6)
- Retrieval (NTCIR-12, ARQMath)
- Semantic similarity (human judgments)
- Zero-shot classification
- Baseline comparisons (TangentCFT, GCL, SciBERT)
- **Deliverable**: Evaluation report

### Phase 5: Applications (Month 7)
- Equation auto-completion demo
- Semantic search interface
- Cross-domain analogy finder
- **Deliverable**: Demo applications

### Phase 6: Paper (Months 7-8)
- Target: NeurIPS, ICLR, ICML
- **Deliverable**: Conference submission

---

## Key Technical Challenges

### Challenge 1: Equation Parsing
- **Problem**: Convert LaTeX to structured representation
- **Solution**: LaTeXML (proven on 1.78M arXiv papers)

### Challenge 2: Data Quality
- **Problem**: Noisy equation-description extraction
- **Solution**: Heuristic filtering + manual curation + leverage MathBridge dataset

### Challenge 3: Domain Diversity
- **Problem**: Generalization across physics subdomains
- **Solution**: Stratified sampling + domain-adversarial training + cross-domain eval

### Challenge 4: Evaluation
- **Problem**: Limited human-annotated similarity data
- **Solution**: Create 1K annotated pairs + leverage existing benchmarks (NTCIR, ARQMath)

### Challenge 5: Computational Cost
- **Problem**: Training on 500K pairs with GNN
- **Solution**: Mixed precision + gradient accumulation + distributed training

---

## Expected Contributions

1. **Novel Method**: First contrastive learning approach for equation-text pairs
2. **Domain Focus**: Physics-specific (vs general mathematics)
3. **Large Dataset**: 500K curated physics equation-description pairs
4. **Strong Baselines**: Compare to TangentCFT, GCL, SciBERT
5. **Comprehensive Eval**: Retrieval + similarity + classification + clustering
6. **Novel Applications**: Auto-completion, semantic search, cross-domain analogies

---

## Success Metrics

**Quantitative Targets**:
- Retrieval Recall@5 > 70% (vs TangentCFT ~50%)
- Semantic Similarity ρ > 0.7
- Zero-shot Classification > 75%
- Clustering ARI > 0.6

**Qualitative Goals**:
- Semantically meaningful equation clusters
- Natural emergence of physics analogies
- Zero-shot transfer to new domains

---

## Critical Resources

### Code
- **TangentCFT**: https://github.com/BehroozMansouri/TangentCFT
- **GCL Formula**: https://github.com/WangPeiSyuan/GCL-Formula-Retrieval
- **SciBERT**: https://github.com/allenai/scibert
- **LaTeXML**: https://github.com/brucemiller/LaTeXML

### Pre-trained Models
- **PhysBERT**: Check arXiv:2408.09574 for release
- **SciBERT**: `allenai/scibert_scivocab_uncased`
- **MathBERTa**: `witiko/mathberta`

### Datasets
- **NTCIR-12**: http://ntcir-math.nii.ac.jp/data/
- **ARQMath**: https://www.cs.rit.edu/~dprl/ARQMath/
- **arXiv**: https://arxiv.org/help/bulk_data

### Key Papers
- **GCL for Math**: Wang & Chen, arXiv:2402.13444 (2024)
- **TangentCFT**: Mansouri et al., ICTIR 2019
- **PhysBERT**: arXiv:2408.09574 (2024)
- **MathBridge**: arXiv:2408.07081 (2024)
- **CLIP**: Radford et al., 2021

---

## Next Steps

1. **Immediate (Week 1)**:
   - Set up arXiv bulk data access
   - Install LaTeXML
   - Download pre-trained PhysBERT/SciBERT

2. **Short-term (Month 1)**:
   - Implement data extraction pipeline
   - Start collecting equation-description pairs
   - Prototype GNN equation encoder

3. **Medium-term (Months 2-3)**:
   - Complete dataset curation
   - Implement full Equation-CLIP
   - Begin training experiments

4. **Research Questions**:
   - GNN vs Tree-LSTM vs Transformer?
   - SLT, OPT, or both?
   - Optimal temperature τ?
   - How much data needed?

---

## Conclusion

**Equation-CLIP is feasible and timely**. The field has the necessary ingredients:
- ✅ State-of-the-art methods (GCL, contrastive learning)
- ✅ Pre-trained models (PhysBERT, SciBERT)
- ✅ Large-scale datasets (arXiv, MathBridge)
- ✅ Robust tools (LaTeXML, GNN libraries)
- ✅ Established benchmarks (NTCIR, ARQMath)

The recommended approach builds on proven components (GNN + PhysBERT + InfoNCE) while introducing novel contributions (physics focus, 500K dataset, comprehensive evaluation).

**Expected timeline**: 7-8 months from data collection to paper submission.

**Target impact**: Enable semantic equation search, cross-domain discovery, and equation understanding at scale for the physics community.
