# Equation Embeddings and Mathematical Retrieval Systems Research

## Research Overview

This document provides comprehensive research on equation embeddings, mathematical retrieval systems, and related technologies for the Equation-CLIP project. The research focuses on state-of-the-art methods, benchmarks, datasets, and architectures relevant to building a contrastive learning system for physics equations.

---

## 1. TangentCFT: State-of-the-Art Mathematical Retrieval

### 1.1 Overview

**Tangent-CFT** (Tangent Combined FastText) is the current state-of-the-art embedding model for mathematical formula retrieval, developed by Mansouri, Rohatgi, Oard, Wu, Giles, and Zanibbi.

**Paper**: "Tangent-CFT: An Embedding Model for Mathematical Formulas" (ICTIR 2019)

**Key Innovation**: Combines structural representations of formulas with fastText embeddings to create holistic vector representations.

### 1.2 Architecture & Approach

#### Dual Hierarchical Representations

TangentCFT uses two complementary tree structures:

1. **Symbol Layout Trees (SLT)**:
   - Captures visual appearance and spatial arrangement
   - Represents LaTeX presentation format
   - Nodes = individual symbols
   - Edges = spatial relationships (superscript, subscript, adjacent)

2. **Operator Trees (OPT)**:
   - Captures mathematical content and semantics
   - Represents Content MathML format
   - Internal nodes = operators
   - Leaf nodes = operands

#### Tuple Generation Process

**Tangent-S Algorithm**:
- Performs depth-first traversal of SLT/OPT trees
- Generates tuples for pairs of symbols: `(s1, s2, R, FRP)`
  - `s1`: ancestor symbol
  - `s2`: descendant symbol
  - `R`: edge label sequence from s1 to s2
  - `FRP`: full relative path

**Example**: For expression `2 + (z - 1)`:
- Tuples capture relationships like `(+, -, [down-left], ...)`
- Each tuple encodes structural path information

#### FastText Embedding

- Tuples encoded as strings and embedded using fastText n-gram model
- FastText captures sub-string similarities
- Formula representation = average of all tuple embeddings
- Final embedding combines SLT + OPT vectors (simple vector sum)

### 1.3 Performance Benchmarks

**NTCIR-12 Results**:
- State-of-the-art performance on formula retrieval task
- ~590,000 formulas from Wikipedia
- Outperformed traditional structural matching methods
- Successfully retrieves partially similar formulas (fuzzy matching)

**Strengths**:
- Fine-grained holistic vector representations
- Retrieves many more partially similar formulas than tree-matching methods
- Combines with Approach0 for hybrid structural+semantic search

**Limitations**:
- Recent Graph Contrastive Learning (GCL) methods now exceed TangentCFT performance (2024)
- Unsupervised approach - doesn't leverage labeled data
- Averaged embeddings may lose fine-grained structural information

### 1.4 Available Implementations

**GitHub Repositories**:
- Primary: `BehroozMansouri/TangentCFT` (Python 3.12)
- Fork: `StellarL/TangentCFT` (Python 3.6)
- Evaluated on NTCIR-12 dataset

---

## 2. Equation Embedding Methods

### 2.1 Krstovski & Blei (2018): "Equation Embeddings"

**Paper**: arXiv:1803.09123 (March 2018)

**Core Idea**: Unsupervised approach discovering semantic representations by leveraging surrounding text context (inspired by word2vec).

#### Key Approaches

**EqEmb (Equations as Single Tokens)**:
- Treats entire equations as atomic tokens
- Learns embeddings using word2vec skip-gram model
- Context window captures relationship with surrounding words

**EqEmb-U (Unpacked Equations)**:
- Treats variables, symbols, operators as individual tokens
- Equations = sequences of tokens
- Equation embedding = average of token embeddings

#### Mathematical Representation

- Uses **Syntax Layout Tree (SLT)** representation
- Captures spatial relationships between symbols
- Based on Zanibbi et al. 2016b

#### Dataset & Evaluation

- 4 arXiv collections: NLP, IR, AI, ML domains
- ~98,500 equations analyzed
- Demonstrates semantic clustering of equations

**Strengths**:
- Fully unsupervised - requires no labeled data
- Leverages abundant unlabeled scientific text
- Contextual understanding from surrounding text

**Limitations**:
- No explicit structural modeling
- Averaged representations may be too coarse
- Limited to domains with sufficient text context

### 2.2 Recent Advances: SSEmb (2024)

**SSEmb**: Joint Structural and Semantic Embedding Framework

**Paper**: arXiv:2508.04162 (August 2024)

- Combines structural tree features with semantic embeddings
- Addresses limitations of structure-only or semantic-only approaches
- Shows improved performance on formula retrieval

---

## 3. MathBERT and Related Pre-trained Models

### 3.1 MathBERT for Formula Understanding (Peng et al., 2021)

**Paper**: arXiv:2105.00377

**First pre-trained model specifically for mathematical formula understanding**

#### Key Features

**Pre-training Tasks**:
- Masked Language Modeling (MLM) for surrounding text
- **Masked Substructure Prediction**: Novel task predicting masked formula substructures from Operator Tree (OPT)
- Joint training on formulas + context

**Architecture**:
- Based on BERT-base
- Custom tokenization for mathematical symbols
- Handles both text and formula tokens

**Applications**:
- Formula retrieval
- Formula classification
- Formula-text matching

### 3.2 MathBERT for Mathematics Education (Shen et al., 2021)

**Paper**: arXiv:2106.07340

**Broader domain coverage from pre-k to graduate level**

#### Training Details

- Pre-trained on 100M tokens of mathematical content
- Curriculum: pre-k, K-12, college, graduate level
- Sources: textbooks, curriculum documents, arXiv abstracts

**Domain**: General mathematics education (not physics-specific)

### 3.3 MathBERTa (Witiko)

**Available on Hugging Face**: `witiko/mathberta`

#### Key Features

- Based on RoBERTa architecture
- Tokenizer extended with LaTeX math symbols
- Large-scale training corpus

**Training Data**:
- ArXMLiv 2020: 1,581,037 arXiv documents
- Math StackExchange: 2,466,080 Q&A pairs
- Total: ~52GB text + LaTeX

**Tokenization**: Extended vocabulary with LaTeX-specific tokens

### 3.4 PhysBERT: Physics-Specific Model (2024)

**Paper**: arXiv:2408.09574 (August 2024)

**First physics-specific text embedding model**

#### Training Approach

- Pre-trained on 1.2M arXiv physics papers
- Supervised fine-tuning on physics tasks
- Deep understanding of physics language

**Performance**: Significantly outperforms general models (including SciBERT) on physics-specific NLP tasks

#### Relevance to Equation-CLIP

**Strong candidate for text encoder**:
- Domain-specific (physics focus)
- Large-scale physics corpus
- Proven performance on physics text

### 3.5 SciBERT (Beltagy et al., 2019)

**Paper**: arXiv:1903.10676

**Baseline pre-trained model for scientific text**

#### Training Details

- Corpus: 1.14M papers (3.1B tokens) from SemanticScholar
- 18% Computer Science, 82% Biomedical
- Full text training (not just abstracts)
- Custom vocabulary (scivocab) built from corpus

**Performance**: State-of-the-art on multiple scientific NLP benchmarks

**GitHub**: `allenai/scibert`

---

## 4. Mathematical Information Retrieval Benchmarks

### 4.1 NTCIR Math Tasks (NTCIR-10, 11, 12)

**Focus**: Techniques for searching mathematical content with formula expressions

#### NTCIR-12 MathIR Task

**Datasets**:

1. **Wikipedia Corpus**:
   - 319,689 articles
   - 592,443 tagged formulas
   - Use case: math retrieval by non-experts

2. **arXiv Corpus**:
   - 105,120 scientific papers
   - ~60M formulas
   - Domains: CS, math, physics, statistics
   - Categories: math, cs, physics:math-ph, stat, physics:hep-th, physics:nlin

**Tasks**:
- Formula Browsing (Task 2): Retrieve relevant formulas given query formula
- Formula + keyword search
- New "simto region" operator for approximate matching

**Participation**: 6 teams, 37 topics

**Data Access**: `http://ntcir-math.nii.ac.jp/data/`

### 4.2 ARQMath Lab (CLEF 2020-2022)

**Focus**: Answer retrieval for math questions from Math Stack Exchange

#### Task Structure

**Task 1: Answer Retrieval**
- Find answers to math questions
- Queries + corpora from Math Stack Exchange
- Text + formulas

**Task 2: Formula Retrieval**
- Similar to NTCIR-12 Wikipedia Formula Browsing
- Contextualized formula search
- 76 queries, avg 63.18 relevant docs per query

**Task 3: Open Domain QA**

#### Evaluation Metrics

**Primary Metrics**:
- **nDCG' (nDCG-prime)**: Excludes unjudged documents
- **MAP' (Mean Average Precision prime)**: H+M binarization
- **P'@10 (Precision at 10 prime)**: H+M binarization

**Additional Metrics**:
- PR'@{15, 20} (precision-recall at different cutoffs)
- P'@5
- AR (Average Relevance) for Task 3: bounded [0, 3]

### 4.3 MIRB: Mathematical Information Retrieval Benchmark (2025)

**Paper**: arXiv:2505.15585 (May 2025)

**Modern comprehensive benchmark consolidating multiple datasets**

#### Task Coverage

1. **Semantic Statement Retrieval**
2. **Question-Answer Retrieval**
3. **Premise Retrieval**
4. **Formula Retrieval**

**Total**: 12 datasets across 4 tasks

**Datasets Included**:
- NTCIR-12 Wikipedia Formula Browsing
- ARQMath-Task 2
- Various semantic similarity datasets

**Significance**: First unified benchmark for comparing MIR systems

### 4.4 Benchmark Comparison

| Benchmark | Source | Task Focus | Corpus Size | Key Metrics |
|-----------|--------|------------|-------------|-------------|
| NTCIR-12 | Wikipedia, arXiv | Formula retrieval | 590K formulas | bpref, nDCG |
| ARQMath | Math StackExchange | Answer + formula retrieval | 2.4M Q&A | nDCG', MAP' |
| MIRB | Multiple sources | 4-task comprehensive | 12 datasets | Task-specific |

---

## 5. Tree-Based Representations for Equations

### 5.1 Abstract Syntax Trees (AST) for Mathematics

**Definition**: Tree structure representing abstract syntactic structure where:
- Interior nodes = operators
- Leaf nodes = operands (variables, constants)
- Edges = hierarchical relationships

**Key Property**: Parentheses are implicit in tree structure

**Example**: `2 + (z - 1)`
```
    +
   / \
  2   -
     / \
    z   1
```

**Advantages**:
- Encodes order of operations
- Describes associativity
- Eliminates need for parentheses
- Operator precedence encoded by tree depth

### 5.2 Symbol Layout Tree (SLT)

**Purpose**: Represents visual/spatial layout of mathematical notation

**Structure**:
- Nodes = individual symbols
- Edges = spatial relationships
  - Superscript
  - Subscript
  - Adjacent (horizontal)
  - Above/below (fractions)

**Format**: Typically from LaTeX presentation markup

**Use Cases**:
- OCR and handwriting recognition
- Visual similarity search
- Layout-aware retrieval

### 5.3 Operator Tree (OPT)

**Purpose**: Represents semantic mathematical content

**Structure**:
- Internal nodes = mathematical operators/functions
- Leaf nodes = operands (variables, numbers)
- Captures mathematical meaning

**Format**: Content MathML representation

**Properties**:
- Canonicalization possible
- Semantic equivalence detection
- Structure reflects mathematical operations

### 5.4 Binary Expression Trees

**Specialized form for binary operators**:
- Each node has 0 or 2 children
- Algebraic expressions: +, -, *, /, ^
- Boolean expressions: AND, OR, NOT

**Traversals**:
- Infix: traditional notation
- Prefix: Polish notation
- Postfix: Reverse Polish notation

### 5.5 LaTeX AST Parsing

**Challenges**:
- LaTeX is less rigid than programming languages
- TeX never generates full document tree
- Macro expansion complexity

**Tools**:
- LaTeXML: Best for AST-like representations
- Custom parsers: `leenty/latex-ast` (GitHub)
- Recursive descent parsing for expressions

### 5.6 Encoding Trees for Neural Networks

**Approaches**:

1. **Flattening**:
   - Pre-order, post-order, in-order traversals
   - Sequence models (RNN, LSTM, Transformer)
   - Loss: hierarchical structure information

2. **Recursive Neural Networks**:
   - Apply same weights recursively over tree
   - Bottom-up composition
   - Preserves structure explicitly

3. **Tree-LSTM**:
   - Extends LSTM to tree structures
   - Child-sum Tree-LSTM
   - N-ary Tree-LSTM

4. **Graph Neural Networks**:
   - Trees as special case of graphs
   - Message passing on tree edges
   - Flexible aggregation schemes

5. **Positional Encodings**:
   - Flatten with position embeddings
   - Tree positional encodings (e.g., Shiv & Quirk 2019)

---

## 6. Neural Network Architectures for Mathematical Expressions

### 6.1 Recursive Neural Networks (Recursive-NN)

**Architecture**: Apply same weights recursively over tree structure

**Process**:
1. Start at leaves (terminals)
2. Apply neural network to combine children
3. Propagate upward to root
4. Root vector = entire expression representation

**Applications**:
- Math word problem solving
- Arithmetic expression evaluation
- Equation scoring for symbolic reasoning

**Papers**:
- "Solving arithmetic word problems by scoring equations with recursive neural networks" (Expert Systems, 2021)
- Tree-LSTM for encoding equation structures

**Advantages**:
- Naturally captures hierarchical structure
- Parameter sharing across tree
- End-to-end differentiable

**Disadvantages**:
- Requires tree structure as input
- Training can be complex
- Less popular than Transformers in modern work

### 6.2 Tree Transformer Models

**Recent Trends**: Adapting Transformers for tree structures

**Approaches**:

1. **Linearization + Positional Encoding**:
   - Flatten tree to sequence
   - Add tree-aware positional encodings
   - Standard Transformer architecture

2. **Modified Attention**:
   - Mask attention based on tree structure
   - Only attend to ancestors/descendants
   - Preserve structural inductive bias

3. **Hybrid Models**:
   - Tree-based encoder
   - Transformer decoder
   - Best of both worlds

**Example Applications**:
- Mathematical expression recognition (handwriting to LaTeX)
- Formula completion
- Symbolic mathematics

### 6.3 Encoder-Decoder for Math

**Standard Architecture** (Deng et al.):

**Encoder**:
- CNN for images (handwritten/printed formulas)
- Feature maps with 2D positional encoding
- Captures spatial relationships

**Decoder**:
- RNN/Transformer with attention
- Generates LaTeX sequence
- Coarse-to-fine attention mechanism

**Datasets**: IM2LATEX-100K

### 6.4 Graph Neural Networks for Formulas

**Recent Innovation**: Treat formulas as graphs

**Advantages**:
- More general than trees (handle equivalences)
- Flexible message passing
- Can incorporate different edge types (spatial, semantic, equivalence)

**State-of-the-Art** (2024):
- Graph Contrastive Learning (GCL) for formula retrieval
- Outperforms TangentCFT
- Self-supervised learning on formula graphs

---

## 7. Graph Contrastive Learning for Mathematical Retrieval

### 7.1 Key Paper (2024)

**Title**: "The Effectiveness of Graph Contrastive Learning on Mathematical Information Retrieval"

**Authors**: Pei-Syuan Wang, Hung-Hsuan Chen

**Paper**: arXiv:2402.13444 (February 2024)

**Published**: ECIR 2024 (European Conference on Information Retrieval)

### 7.2 Core Approach

**Problem Statement**:
- Model needs to capture notation structure
- Lack of relevance scores between formula pairs (unsupervised setting)

**Solution**: Self-supervised Graph Contrastive Learning (GCL)

#### GCL Methods Explored

1. **InfoGraph**: Maximizes mutual information between graph-level and node-level representations

2. **GraphCL**: Uses augmentations (node dropping, edge perturbation, subgraph sampling)

3. **BGRL** (Bootstrapped Graph Representation Learning): Bootstrap approach without negative samples

#### Graph Structures

**Symbol Layout Tree (SLT)** as graph:
- Nodes = symbols
- Edges = spatial relationships

**Operator Tree (OPT)** as graph:
- Nodes = operators/operands
- Edges = semantic relationships

### 7.3 Performance Results

**Key Finding**: GCL consistently exceeds TangentCFT performance

**Metrics**:
- **bpref**: Both metrics show GCL superior
- **nDCG**: Consistently better than TangentCFT
- **Stability**: Very stable results across runs

**Benchmarks**: NTCIR-12 Wikipedia Formula Browsing task

### 7.4 Why GCL Works

**Advantages**:
1. **Self-supervised**: No labeled similarity data needed
2. **Structure-aware**: Learns from graph topology
3. **Holistic representations**: Captures global formula properties
4. **Contrastive objective**: Discriminative embeddings

**Comparison to TangentCFT**:
- TangentCFT: Hand-crafted tuple features + fastText
- GCL: End-to-end learned representations
- GCL captures patterns TangentCFT misses

### 7.5 Implementation

**Code**: `https://github.com/WangPeiSyuan/GCL-Formula-Retrieval`

**Open source**: Full implementation available

---

## 8. Contrastive Learning Fundamentals

### 8.1 Core Principle

**Goal**: Learn embedding space where:
- Similar samples are close together
- Dissimilar samples are far apart

**Training**: Use positive pairs (similar) and negative pairs (dissimilar)

### 8.2 Loss Functions

#### InfoNCE Loss

**Most common formulation**:

```
L = -log(exp(sim(anchor, positive)/τ) / Σ_i exp(sim(anchor, negative_i)/τ))
```

Where:
- `sim()` = similarity function (cosine similarity)
- `τ` = temperature parameter
- Denominator sums over all negatives

**Properties**:
- Encourages anchor-positive similarity
- Pushes apart anchor-negative pairs
- Temperature controls distribution peakiness

#### NT-Xent Loss

**Normalized Temperature-scaled Cross Entropy**:
- Modification of N-pair loss
- Adds temperature parameter
- Used in SimCLR

#### Contrastive Loss (Binary)

```
L = (1-y) * d² + y * max(0, margin - d)²
```

Where:
- `d` = distance between embeddings
- `y` = label (1 if similar, 0 if dissimilar)
- `margin` = threshold for dissimilar pairs

### 8.3 Temperature Parameter τ

**Typical value**: 0.07 (CLIP uses this)

**Effect**:
- Low τ (e.g., 0.01): Sharp distribution, focuses on hard negatives
- High τ (e.g., 0.5): Smooth distribution, considers all negatives equally

**Tuning**: Important hyperparameter for performance

### 8.4 Positive and Negative Pairs

**Positive Pairs** (Equation-CLIP context):
- (equation, its description)
- (equation, related equation)
- Augmented versions of same equation

**Negative Pairs**:
- (equation, unrelated description)
- Equations from different physics domains
- Hard negatives: similar but distinct equations

### 8.5 Physics-Informed Contrastive Learning (PICL)

**Paper**: "PICL: Physics informed contrastive learning for partial differential equations" (APL Machine Learning, 2024)

**Innovation**: Generalized Contrastive Loss (GCL)

**Key Idea**:
- Use governing equation coefficients as similarity measure
- Similar physics → similar embeddings
- Preserves physical laws in embedding space

**Formula**:
```
L_GCL = L_similar + L_dissimilar
```

Where similarity is defined by physics parameters, not just labels.

**Relevance**: Shows contrastive learning can embed physical semantics

---

## 9. Datasets and Data Sources

### 9.1 ArXiv Physics Papers

**Scale**: Millions of papers with LaTeX source

**Access**: arXiv API, bulk data access

**Extraction Tools**:
- **LaTeXML**: Convert LaTeX to HTML + MathML
  - Processed 1.78M arXiv documents (as of 2024)
  - Experimental service promoted to main site
  - Handles 90% of documents (60% without errors)

- **HopTeX**: Large-scale LaTeX processing
  - GitHub: `hopper-project/hoptex`
  - Batch conversion for entire arXiv corpus
  - Extracts equations in MathML format

**Physics Domains**:
- math-ph: Mathematical physics
- hep-th: High-energy physics (theory)
- cond-mat: Condensed matter
- quant-ph: Quantum physics
- astro-ph: Astrophysics
- physics (general)

### 9.2 MathBridge Dataset (2024)

**Paper**: arXiv:2408.07081 (August 2024)

**Focus**: Mathematical spoken sentences → LaTeX formulas

**Scale**:
- 23M LaTeX formulas with spoken descriptions
- 13M unique formulas
- Sources: 48M formulas from arXiv + 1M from textbooks

**Relevance**: Large-scale (formula, description) pairs

**Limitation**: Spoken/procedural descriptions (not conceptual physics descriptions)

### 9.3 OpenWebMath

**Focus**: High-quality mathematical web content

**Sources**:
- Q&A forums (Math Stack Exchange, Physics Forums)
- Educational documents
- Blogs and wikis

**Domains**: Math, physics, CS, technical fields

**Content**: Text + LaTeX extraction, boilerplate removal

**Relevance**: Diverse mathematical content with context

### 9.4 Speech-to-LaTeX Datasets

**S2L Dataset**:
- 66K human-annotated samples
- 571K TTS-generated samples
- Equations + sentences
- Extracted from arXiv (Proof-Pile-2)

**Gr2Tex (Greek2MathTex)**:
- 500 equation pairs
- Natural text ↔ LaTeX notation
- Greek language focus

### 9.5 Wikipedia Mathematical Content

**NTCIR-12 Wikipedia Collection**:
- 319,689 articles
- 592,443 formulas
- General mathematics (not physics-specific)

**Extraction**: LaTeXML, custom parsers

### 9.6 Textbooks

**Advantages**:
- High-quality curation
- Clear explanations
- Pedagogical structure

**Challenges**:
- Copyright restrictions
- Smaller scale than arXiv
- OCR for older texts

### 9.7 Data Augmentation Strategies

**For Equations**:
- Variable renaming (x → y)
- Operator variation (sum → integral)
- Simplification/expansion
- Unit changes
- Domain transfer (1D → 2D)

**For Text**:
- Paraphrasing
- Back-translation
- Synonym replacement
- Sentence reordering

---

## 10. Evaluation Metrics for Equation-CLIP

### 10.1 Retrieval Metrics

#### Recall@K

**Definition**: Fraction of relevant items in top-K results

```
Recall@K = (# relevant in top-K) / (# total relevant)
```

**Typical values**: K = 1, 5, 10, 20

**Target for Equation-CLIP**: Recall@5 > 70%

#### Mean Reciprocal Rank (MRR)

**Definition**: Average of reciprocal ranks of first relevant item

```
MRR = (1/|Q|) * Σ (1 / rank_i)
```

**Range**: [0, 1], higher is better

**Interpretation**: How high is the first relevant result?

#### Normalized Discounted Cumulative Gain (NDCG)

**Definition**: Position-aware relevance metric

```
DCG@K = Σ (rel_i / log2(i+1))
NDCG@K = DCG@K / IDCG@K
```

Where:
- `rel_i` = relevance score of item at position i
- `IDCG` = ideal DCG (best possible ranking)

**Range**: [0, 1], higher is better

**Advantage**: Accounts for graded relevance, position matters

#### Precision@K

**Definition**: Fraction of relevant items in top-K

```
P@K = (# relevant in top-K) / K
```

**Simpler than Recall**: Doesn't require knowing total relevant

### 10.2 Semantic Similarity Metrics

#### Spearman Rank Correlation (ρ)

**Purpose**: Correlation between predicted and human similarity judgments

**Computation**:
1. Humans rate equation pairs for similarity (1-5 scale)
2. Model computes cosine similarity of embeddings
3. Compute Spearman correlation

**Range**: [-1, 1], higher is better

**Target**: ρ > 0.7

**Dataset Requirements**: Human-annotated similarity pairs

### 10.3 Zero-Shot Classification

#### Task

Given equation embedding, predict physics domain:
- Classical mechanics
- Electromagnetism
- Quantum mechanics
- Thermodynamics
- Relativity
- Fluid dynamics

#### Metrics

**Accuracy**: Simple classification accuracy

```
Acc = (# correct predictions) / (# total)
```

**F1 Score**: Harmonic mean of precision and recall

**Target**: Accuracy > 75%

#### Evaluation Protocol

1. Train linear classifier on frozen embeddings
2. Or: Use text descriptions as class prototypes
3. Zero-shot: No labeled equation examples in training

### 10.4 Clustering Metrics

#### Adjusted Rand Index (ARI)

**Purpose**: Measures similarity to ground truth clustering

**Properties**:
- Range: [-1, 1]
- 0 = random clustering
- 1 = perfect agreement
- Adjusted for chance

**Computation**: Based on pair-wise agreement

**Target**: ARI > 0.6

#### Normalized Mutual Information (NMI)

**Purpose**: Measures shared information between clusterings

**Properties**:
- Range: [0, 1]
- 0 = independent
- 1 = identical
- Not adjusted for chance

**Computation**: Based on entropy and mutual information

**Typical Values**: 0.4-0.7 for good clustering

#### Silhouette Score

**Purpose**: Measures cluster cohesion and separation (intrinsic)

**Properties**:
- Range: [-1, 1]
- -1 = incorrect clustering
- 0 = overlapping clusters
- 1 = dense and well-separated

**Computation**: For each point, compare average intra-cluster distance to average nearest-cluster distance

**Advantage**: Doesn't require ground truth labels

**Target**: Silhouette > 0.5

### 10.5 Extrinsic vs Intrinsic Metrics

**Extrinsic** (require ground truth):
- ARI
- NMI
- Accuracy
- Retrieval metrics (with relevance judgments)

**Intrinsic** (no ground truth needed):
- Silhouette score
- Calinski-Harabasz index
- Davies-Bouldin index

### 10.6 Baseline Comparisons

**Essential baselines for Equation-CLIP**:

1. **TF-IDF + Edit Distance**:
   - String similarity baseline
   - LaTeX as text

2. **TangentCFT**:
   - Current MIR state-of-the-art
   - NTCIR-12: ~50% Recall@5

3. **Graph Contrastive Learning**:
   - 2024 state-of-the-art
   - Outperforms TangentCFT

4. **SciBERT (text-only)**:
   - Text descriptions only
   - No equation encoder

5. **Krstovski & Blei**:
   - Unsupervised equation embeddings
   - Context-based

---

## 11. Key Researchers and Papers

### 11.1 Mathematical Information Retrieval

**Richard Zanibbi**:
- Rochester Institute of Technology
- TangentCFT co-author
- Pioneer in math retrieval
- Key papers: Tangent-CFT, Symbol Layout Trees

**Douglas W. Oard**:
- University of Maryland
- Information retrieval expert
- ARQMath organizer

**Behrooz Mansouri**:
- TangentCFT primary author
- GitHub: BehroozMansouri/TangentCFT

**Pei-Syuan Wang & Hung-Hsuan Chen**:
- Graph Contrastive Learning for math retrieval (2024)
- State-of-the-art results

### 11.2 Equation Embeddings

**Kriste Krstovski & David M. Blei**:
- "Equation Embeddings" (2018)
- Unsupervised contextual embeddings
- Columbia University / Princeton

### 11.3 Mathematical Language Models

**Shen et al.**:
- MathBERT for education (2021)
- Large-scale math corpus

**Peng et al.**:
- MathBERT for formula understanding (2021)
- First pre-trained model for formulas

**PhysBERT Team**:
- Physics-specific BERT (2024)
- 1.2M arXiv physics papers

### 11.4 Deep Learning for Math

**Andrew Lan** (UMass Amherst):
- Tree embeddings for formulas
- Math education applications
- GitHub: umass-ml4ed/mathGPT

**François Charton & Guillaume Lample** (Meta):
- Transformers for symbolic math
- "Deep Learning for Symbolic Mathematics"

### 11.5 Contrastive Learning

**Radford et al. (OpenAI)**:
- CLIP: Contrastive Language-Image Pre-training (2021)
- Foundation for multimodal contrastive learning

**Ting Chen et al. (Google)**:
- SimCLR (2020)
- Self-supervised contrastive learning

**Lilian Weng**:
- Comprehensive blog on contrastive learning
- Tutorial: lilianweng.github.io

---

## 12. Recommendations for Equation-CLIP Architecture

### 12.1 Equation Encoder

**Recommended Approach**: Hybrid Tree-Graph Architecture

#### Option A: Graph Neural Network (Preferred)

**Rationale**:
- Recent GCL methods outperform TangentCFT
- Flexible representation
- Can incorporate multiple edge types (spatial + semantic)

**Architecture**:
1. Parse LaTeX to SLT + OPT
2. Combine into unified graph
3. Graph Convolutional Network (GCN) or Graph Attention Network (GAT)
4. Graph-level pooling (e.g., global mean/max)
5. MLP projection head

**Reference**: Wang & Chen (2024) GCL for formula retrieval

#### Option B: Tree Transformer (Alternative)

**Architecture**:
1. Parse LaTeX to OPT
2. Serialize with tree-aware positional encodings
3. Transformer encoder (6 layers)
4. CLS token or mean pooling
5. MLP projection head

**Advantage**: Leverages pre-trained Transformers

#### Option C: Dual-Stream (Hybrid)

**Architecture**:
1. **Structural stream**: GNN on OPT
2. **Sequential stream**: Transformer on linearized LaTeX
3. Combine outputs (concatenate or attention fusion)
4. MLP projection head

**Advantage**: Best of both worlds

### 12.2 Text Encoder

**Recommended**: PhysBERT or SciBERT

#### PhysBERT (Preferred for Physics)

**Rationale**:
- Domain-specific: 1.2M arXiv physics papers
- Outperforms SciBERT on physics tasks
- Deep understanding of physics terminology

**Architecture**:
- Fine-tune on (equation, description) pairs
- Keep early layers frozen, fine-tune last 3-6 layers
- Add projection head

#### SciBERT (Fallback)

**Rationale**:
- Well-established baseline
- Broader scientific coverage
- Extensive pre-training

**Architecture**: Same as PhysBERT

#### Comparison

| Model | Domain | Training Data | Performance | Availability |
|-------|--------|---------------|-------------|--------------|
| PhysBERT | Physics | 1.2M physics papers | Best for physics | Newer (2024) |
| SciBERT | General science | 1.14M papers (18% CS, 82% bio) | Strong baseline | Well-established |
| MathBERT | Math formulas | Math + formula datasets | Formula-specific | 2021 |

### 12.3 Projection Heads

**Architecture**: 2-layer MLP

```
Input → Linear(dim → 512) → ReLU → Linear(512 → 256) → L2 Normalize
```

**Output**: 256-dimensional normalized embeddings

**Why 256-d**:
- CLIP uses 512-d for images, but equations are simpler
- 256-d balances expressiveness and efficiency
- Can experiment with 128, 256, 512

### 12.4 Loss Function

**Recommended**: InfoNCE Loss (CLIP-style)

```python
def infonce_loss(equation_embs, text_embs, temperature=0.07):
    # Normalize
    equation_embs = F.normalize(equation_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)

    # Similarity matrix
    logits = torch.matmul(equation_embs, text_embs.T) / temperature

    # Contrastive loss (both directions)
    labels = torch.arange(len(logits)).to(logits.device)
    loss_eq = F.cross_entropy(logits, labels)
    loss_text = F.cross_entropy(logits.T, labels)

    return (loss_eq + loss_text) / 2
```

**Temperature**: Start with τ = 0.07 (CLIP default), tune in [0.01, 0.1]

### 12.5 Training Strategy

#### Curriculum Training

1. **Warm-up** (Epochs 1-10):
   - High-quality textbook equations
   - Clear descriptions
   - Smaller batch size (128)

2. **Main Training** (Epochs 11-50):
   - Full arXiv dataset
   - Batch size 512-1024
   - Hard negative mining

3. **Fine-tuning** (Epochs 51-60):
   - Difficult examples
   - Domain-specific subsets
   - Lower learning rate

#### Hard Negative Mining

**Strategy**: Sample negatives from same physics domain

**Example**:
- Anchor: Maxwell's equation
- Positive: "Describes relationship between electric and magnetic fields"
- Hard negative: Gauss's law equation (related but different)
- Easy negative: Schrodinger equation (different domain)

**Implementation**:
- Online mining: Compute similarities, sample hard negatives each batch
- Offline mining: Pre-compute embeddings, select hard negatives

### 12.6 Data Augmentation

#### Equation Augmentations

1. **Variable renaming**: `F = ma` → `F = m*a'`
2. **Notation variations**: `∂/∂t` → `d/dt`
3. **Algebraic manipulation**: `E = mc²` → `m = E/c²`
4. **Unit changes**: CGS ↔ SI
5. **Dimensional variations**: 1D → 2D → 3D

#### Text Augmentations

1. **Paraphrasing**: "describes force" → "expresses force"
2. **Synonym replacement**: "velocity" → "speed"
3. **Back-translation**: English → German → English
4. **Sentence reordering**
5. **Dropout**: Randomly drop words

**Augmentation probability**: 0.3-0.5 per sample

### 12.7 Hyperparameters

**Recommended Starting Point**:

```yaml
# Model
equation_encoder: GNN  # or TreeTransformer
text_encoder: PhysBERT
embedding_dim: 256
projection_hidden_dim: 512

# Training
batch_size: 512
learning_rate: 3e-4
weight_decay: 0.01
optimizer: AdamW
temperature: 0.07
epochs: 60

# Data
train_size: 500K equation-text pairs
augmentation_prob: 0.4
hard_negative_ratio: 0.3

# Curriculum
warmup_epochs: 10
warmup_lr: 1e-5
main_lr: 3e-4
finetune_lr: 1e-5
```

### 12.8 Evaluation Protocol

**Test Sets**:

1. **Retrieval**:
   - NTCIR-12 Wikipedia formulas
   - ARQMath-2 formula retrieval
   - Custom physics equation retrieval set

2. **Semantic Similarity**:
   - Create human-annotated physics equation pairs
   - Target: 500-1000 pairs
   - 5-point similarity scale

3. **Zero-Shot Classification**:
   - Label equations by physics subdomain
   - 6-10 domains
   - ~1000 test equations

4. **Clustering**:
   - Cluster test set equations
   - Compare to domain labels
   - Visualize with t-SNE/UMAP

**Baselines**:
- TangentCFT
- GCL (Wang & Chen 2024)
- SciBERT (text-only)
- TF-IDF + Edit Distance

---

## 13. Technical Challenges and Mitigation Strategies

### 13.1 Equation Representation Challenge

**Problem**: How to encode hierarchical tree structure effectively?

**Solutions**:
1. **Use GNNs**: Proven effective (Wang & Chen 2024)
2. **Dual representation**: Combine tree + sequence
3. **Pre-training**: Self-supervised pre-training on formulas alone
4. **Ablation study**: Compare tree vs sequence vs hybrid

### 13.2 Data Quality Challenge

**Problem**: Extracting high-quality (equation, description) pairs from papers

**Solutions**:
1. **LaTeXML**: Robust LaTeX → MathML conversion
2. **Heuristic filtering**:
   - Equations with captions
   - Equations in definition environments
   - Equations followed by "where..." clauses
3. **Manual curation**: Sample and verify
4. **Active learning**: Human-in-the-loop labeling
5. **Leverage MathBridge**: 13M formula-description pairs

### 13.3 Domain Diversity Challenge

**Problem**: Ensuring model generalizes across physics subdomains

**Solutions**:
1. **Stratified sampling**: Balance domains in training
2. **Domain-adversarial training**: Encourage domain-invariant features
3. **Multi-task learning**: Predict domain as auxiliary task
4. **Cross-domain evaluation**: Test on held-out domains
5. **Data augmentation**: Cross-domain analogy transfer

### 13.4 Evaluation Challenge

**Problem**: Creating meaningful test sets with expert annotations

**Solutions**:
1. **Leverage existing benchmarks**: NTCIR-12, ARQMath
2. **Crowdsourcing**: Physics students for annotations
3. **Expert validation**: Physics professors for quality control
4. **Automatic metrics**: Use as proxies (edit distance, tree similarity)
5. **Qualitative analysis**: Case studies, error analysis

### 13.5 Computational Challenge

**Problem**: Large-scale training on 500K+ pairs

**Solutions**:
1. **Efficient architectures**: GNN more efficient than Transformers on trees
2. **Mixed precision training**: FP16 for speed
3. **Gradient accumulation**: Simulate large batch sizes
4. **Distributed training**: Multi-GPU data parallelism
5. **Caching**: Pre-compute equation graphs to avoid re-parsing

### 13.6 Cold-Start Challenge

**Problem**: Zero-shot transfer to completely new physics domains

**Solutions**:
1. **Rich text encoder**: PhysBERT captures broad physics knowledge
2. **Compositional structure**: Learn primitive concepts that generalize
3. **Few-shot fine-tuning**: Quick adaptation with 10-100 examples
4. **Meta-learning**: Learn to learn from few examples
5. **Textbook-first curriculum**: Start with pedagogical content

---

## 14. Implementation Roadmap

### 14.1 Phase 1: Data Collection (Months 1-2)

**Tasks**:
1. Set up arXiv bulk data access
2. Deploy LaTeXML conversion pipeline
3. Extract equations + surrounding text
4. Heuristic filtering for quality
5. Create train/val/test splits
6. Manual annotation of 1000 test pairs

**Deliverables**:
- 100K-500K training pairs
- 2K validation pairs
- 2K test pairs (with annotations)

### 14.2 Phase 2: Model Development (Months 3-4)

**Tasks**:
1. Implement GNN equation encoder
2. Load pre-trained PhysBERT
3. Implement projection heads
4. Implement InfoNCE loss
5. Data augmentation pipeline
6. Training loop with logging

**Deliverables**:
- Working Equation-CLIP implementation
- Baseline models (TangentCFT, SciBERT)

### 14.3 Phase 3: Training (Month 5)

**Tasks**:
1. Hyperparameter tuning
2. Curriculum training
3. Hard negative mining
4. Ablation studies
5. Multi-domain training

**Deliverables**:
- Trained Equation-CLIP model
- Training curves and analysis

### 14.4 Phase 4: Evaluation (Month 6)

**Tasks**:
1. Retrieval evaluation (NTCIR-12, ARQMath)
2. Semantic similarity (human judgments)
3. Zero-shot classification
4. Clustering analysis
5. Baseline comparisons
6. Qualitative analysis

**Deliverables**:
- Comprehensive evaluation report
- Performance comparison tables
- Error analysis

### 14.5 Phase 5: Applications (Month 7)

**Tasks**:
1. Equation auto-completion demo
2. Semantic search interface
3. Cross-domain analogy finder
4. Visualization (t-SNE, UMAP)
5. Case studies

**Deliverables**:
- Demo applications
- User study results

### 14.6 Phase 6: Paper Writing (Month 7-8)

**Target Venues**: NeurIPS, ICLR, ICML (or workshops)

**Sections**:
1. Introduction & Motivation
2. Related Work (TangentCFT, CLIP, GCL)
3. Method (Architecture, Training, Data)
4. Experiments (Retrieval, Similarity, Classification)
5. Ablation Studies
6. Applications
7. Discussion & Future Work

---

## 15. Key Insights and Takeaways

### 15.1 What Works Well

1. **Graph representations**: GCL (2024) beats TangentCFT
2. **Dual representations**: Combine SLT (visual) + OPT (semantic)
3. **Contrastive learning**: InfoNCE loss proven effective (CLIP)
4. **Domain-specific pre-training**: PhysBERT > SciBERT for physics
5. **Large-scale data**: arXiv provides millions of equations
6. **Tree structures**: Capture mathematical hierarchy

### 15.2 What Needs Improvement

1. **Semantic similarity datasets**: Limited human-annotated data
2. **Equation-text pairs**: Automated extraction is noisy
3. **Tree encoding**: No consensus on best neural architecture
4. **Cross-domain transfer**: Limited work on physics-specific retrieval
5. **Evaluation metrics**: MIR metrics not fully standardized

### 15.3 Novel Contributions for Equation-CLIP

1. **First contrastive learning** for equation-text pairs
2. **Physics-specific focus** (vs general math)
3. **Hybrid architecture**: GNN + PhysBERT
4. **Large-scale dataset**: 500K physics equation-description pairs
5. **Comprehensive evaluation**: Retrieval + similarity + classification + clustering
6. **Novel applications**: Auto-completion, cross-domain analogies

### 15.4 Potential Failure Modes

1. **Overfitting to LaTeX syntax**: Model memorizes notation rather than semantics
   - *Mitigation*: Augment notation variations

2. **Domain bias**: Overfit to classical mechanics (most common)
   - *Mitigation*: Stratified sampling, domain-adversarial training

3. **Spurious correlations**: Learn from surrounding text rather than equations
   - *Mitigation*: Ablation study with equation-only encoder

4. **Notation ambiguity**: Same symbol means different things
   - *Mitigation*: Context-aware encoding, use surrounding text

5. **Computational cost**: GNN training on large graphs
   - *Mitigation*: Graph coarsening, efficient implementations

---

## 16. Resources and Links

### 16.1 Code Repositories

- **TangentCFT**: https://github.com/BehroozMansouri/TangentCFT
- **GCL Formula Retrieval**: https://github.com/WangPeiSyuan/GCL-Formula-Retrieval
- **SciBERT**: https://github.com/allenai/scibert
- **LaTeXML**: https://github.com/brucemiller/LaTeXML
- **HopTeX**: https://github.com/hopper-project/hoptex
- **MathGPT**: https://github.com/umass-ml4ed/mathGPT

### 16.2 Pre-trained Models (Hugging Face)

- **SciBERT**: `allenai/scibert_scivocab_uncased`
- **MathBERTa**: `witiko/mathberta`
- **MathBERT**: `tbs17/MathBERT`
- **PhysBERT**: (Check arXiv paper for release)

### 16.3 Datasets

- **NTCIR-12**: http://ntcir-math.nii.ac.jp/data/
- **ARQMath**: https://www.cs.rit.edu/~dprl/ARQMath/
- **MathBridge**: (arXiv:2408.07081)
- **OpenWebMath**: https://huggingface.co/datasets/open-web-math
- **arXiv**: https://arxiv.org/help/bulk_data

### 16.4 Key Papers

#### Mathematical Retrieval
- Tangent-CFT (Mansouri et al., ICTIR 2019)
- GCL for Math Retrieval (Wang & Chen, arXiv:2402.13444, 2024)
- Mathematical Information Retrieval (survey, ACM Computing Surveys, 2024)

#### Equation Embeddings
- Equation Embeddings (Krstovski & Blei, arXiv:1803.09123, 2018)
- SSEmb (arXiv:2508.04162, 2024)

#### Language Models
- MathBERT (Peng et al., arXiv:2105.00377, 2021)
- PhysBERT (arXiv:2408.09574, 2024)
- SciBERT (Beltagy et al., arXiv:1903.10676, 2019)

#### Contrastive Learning
- CLIP (Radford et al., 2021)
- PICL (APL Machine Learning, 2024)

### 16.5 Benchmarks

- **NTCIR-12 MathIR**: http://ntcir-math.nii.ac.jp/
- **ARQMath**: https://www.cs.rit.edu/~dprl/ARQMath/
- **MIRB**: arXiv:2505.15585 (2025)

### 16.6 Tools

- **LaTeXML**: LaTeX → MathML conversion
- **Approach0**: Math search engine (https://approach0.xyz)
- **SymPy**: Symbolic mathematics in Python
- **PyTorch Geometric**: GNN library
- **Hugging Face Transformers**: Pre-trained language models

---

## 17. Research Questions for Further Investigation

### 17.1 Architecture Questions

1. Which tree encoding is best: GNN vs Tree-LSTM vs Transformer?
2. Should we use SLT, OPT, or both?
3. What graph pooling strategy works best?
4. How important is structure vs sequence representation?

### 17.2 Training Questions

1. What's the optimal temperature τ?
2. How many hard negatives per batch?
3. Is curriculum learning necessary?
4. What augmentations help most?
5. How much data is needed for convergence?

### 17.3 Evaluation Questions

1. What's the correlation between retrieval metrics and downstream tasks?
2. Can we predict equation properties from embeddings?
3. How well do embeddings capture physical intuition?
4. What domains generalize best?

### 17.4 Application Questions

1. Can equation auto-completion assist scientists?
2. Does semantic search outperform keyword search?
3. Can we discover novel cross-domain analogies?
4. Can we generate explanations from embeddings?

---

## Conclusion

This research reveals a rapidly evolving field with significant recent advances:

1. **Graph Contrastive Learning (2024)** has surpassed TangentCFT as the state-of-the-art for mathematical retrieval
2. **PhysBERT (2024)** provides physics-specific language understanding
3. **Large-scale datasets** (MathBridge, OpenWebMath) enable training at scale
4. **Tree/graph representations** effectively capture mathematical structure
5. **Contrastive learning** (CLIP-style) is a proven approach for multimodal embeddings

**Equation-CLIP is feasible and timely**. The field has matured to the point where combining these advances into a physics-focused contrastive learning system represents a natural and promising next step.

The recommended architecture (GNN equation encoder + PhysBERT text encoder + InfoNCE loss) builds on proven components while introducing novel contributions: physics specificity, large-scale training, and comprehensive evaluation.

**Key success factors**:
1. High-quality data curation from arXiv
2. Robust equation parsing (LaTeXML)
3. Careful hyperparameter tuning
4. Comprehensive evaluation on multiple benchmarks
5. Novel applications demonstrating practical value

**Expected impact**: If successful, Equation-CLIP could enable semantic equation search, cross-domain discovery, and equation understanding at scale—transforming how scientists interact with mathematical knowledge.
