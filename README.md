# Uncovering Semantic Hierarchy in Text-Attributed Graphs via Large Language Models

This repository contains the official implementation of the paper:

> **Uncovering Semantic Hierarchy in Text-Attributed Graphs via Large Language Models** 

SHiFT automatically uncovers multi-level semantic concepts using LLM-powered hierarchical clustering, and injects the resulting taxonomy into graph representation learning through a three‑part hierarchy-aware objective, including: Taxonomy Partition Alignment, Taxonomy Skeleton Alignment, and Taxonomy Concept Alignment . By iteratively refining the taxonomy and embeddings in an EM-style loop, SHiFT produces not only hierarchy-aware node embeddings, but also human-readable taxonomies with concept names and definitions.

---

## Project Structure

```
.
├── train.py # Main training script
├── train_with_config.py # Training with YAML config
├── dataloader.py # Dataset loading utilities
├── gnn.py # GNN encoder implementations
├── lm.py # Text encoder implementations
├── taxonomy.py # Taxonomy data structures and utilities
├── taxonomy_initialize.py # Cold-start taxonomy construction (E-step)
├── taxonomy_update.py # Incremental taxonomy update (Mapping & Diagnosis)
├── utils.py # Utility functions
├── configs/ # Configuration files
│ └── cora.yaml
├── datasets/ # Dataset directory
├── llm_cache/ # Cached LLM responses
├── output/ # Training outputs and artifacts
└── README.md
````

---

## Installation

```bash
# Create environment
conda create -n htag python=3.9
conda activate htag

# Install PyTorch (adjust CUDA if needed)
pip install torch==2.0.0 torchvision torchaudio \
 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Other dependencies
pip install transformers scikit-learn numpy scipy tqdm \
 openai json-repair pyyaml matplotlib networkx pygraphviz
````

---

## Quick Start

### 1. Prepare Dataset

Download datasets and unzip into `datasets/`:

* [Google Drive](https://drive.google.com/file/d/14GmRVwhP1pUD_OIhoJU3oATZWTnklhPG/view)
* [HuggingFace](https://huggingface.co/datasets/xxwu/LLMNodeBed/tree/main)

Each dataset should follow the PyG format and include:

- `x`: Node features (will be overwritten by text embeddings)
- `edge_index`: Graph structure
- `y`: Node labels (optional, for evaluation)
- `raw_texts`: List of raw text content for each node

---

### 2. Run Training

**Using configuration file (recommended):**

```bash
python train_with_config.py --config configs/cora.yaml
```

---

## Key Components

### 1. Warmup Pretraining

```python
# Structure-aware contrastive learning
learner.warmup_pretrain()
```

Trains the GNN encoder using graph contrastive learning with:
- Edge dropping augmentation
- Feature masking augmentation
- Structure-aware positive/negative sampling

### 2. Taxonomy Construction (E‑Step)

```python
# Build initial taxonomy with LLM
learner.build_initial_taxonomy()
```

Constructs a hierarchical semantic taxonomy using:
- Spherical K-means over-clustering
- Core-set sampling (PageRank, Degree, or Centroid-based)
- LLM-guided concept extraction and refinement

### 3. Hierarchy-Aware Training (M‑Step)

```python
# Train with hierarchical constraints
learner.hierarchy_aware_training()
```

Three complementary losses:

**TPA (Taxonomy Concept Alignment):**
- Treats documents in same leaf cluster as positive pairs
- Enhances local cohesion within fine-grained concepts

**TSA (Taxonomy Skeleton Alignment):**
- Aligns embedding space geometry with taxonomy structure
- Uses Cophenetic Correlation Coefficient (CCC)
- Ensures hierarchical relationships are preserved

**TCA (Taxonomy Concept Alignment):**
- Aligns node embeddings with LLM-generated topic descriptions
- Multi-scale alignment across all hierarchy levels
- Layer-weighted loss (deeper levels have higher weights)

### 4. Incremental Taxonomy Update (E‑Step)

```python
# Update taxonomy every N epochs
# Automatically called during training
```

Efficient surgical interventions:
- **Centroid Re-alignment**: Update cluster centers with new embeddings
- **Global Re-assignment**: Reassign documents based on cosine similarity
* **Split**: low cohesion clusters
* **Merge**: redundant sibling concepts
* **Redefine**: concept drift
* **Reassign**: outlier nodes

---

## Outputs

After training, results are saved in `output/{dataset}/`:

```
output/cora/
├── model_final.pt # Trained model checkpoint
├── learned_embeddings.pt # Final node embeddings
├── taxonomy_initial.pkl # Initial taxonomy (after warmup)
├── taxonomy_initial.json # Human-readable initial taxonomy
├── taxonomy_final.pkl # Final taxonomy (after training)
├── taxonomy_final.json # Human-readable final taxonomy
├── assignments_initial.tsv # Initial document-to-concept assignments
├── assignments_final.tsv # Final document-to-concept assignments
├── taxonomy_epoch_*.pkl # Intermediate taxonomies (if saved)
└── tsne_visualization.png # t-SNE visualization
```

Taxonomies are human-readable and directly usable for analysis.

---

## Taxonomy Exploration

### View Taxonomy Structure

```python
from taxonomy import TaxonomyLoader, TaxonomyVisualizer

# Load taxonomy
root = TaxonomyLoader.from_pickle('output/cora/taxonomy_final.pkl')

# Print tree structure
TaxonomyVisualizer.print_tree(root, max_depth=3)
```

### Query Taxonomy

```python
from taxonomy import TaxonomyLoader, TaxonomyQuery

# Load taxonomy
root = TaxonomyLoader.from_pickle('output/cora/taxonomy_final.pkl')

# Create query interface
query = TaxonomyQuery(root)

# Get statistics
stats = query.get_statistics()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Leaf nodes: {stats['leaf_nodes']}")
print(f"Max depth: {stats['max_depth']}")

# Find concept by name
concept = query.find_concept("Neural Networks")
if concept:
 print(f"Found: {concept.name}")
 print(f"Definition: {concept.definition}")
 print(f"Documents: {len(concept.doc_indices)}")

# Search by keyword
results = query.search_by_keyword("learning")
for node in results:
 print(f"- {query.get_path(node)}")
```

### Export Taxonomy

```bash
# Export to various formats
python -m taxonomy convert \
 --input output/cora/taxonomy_final.pkl \
 --input-format pickle \
 --output output/cora/taxonomy.json \
 --output-format json

# Generate visualization
python -m taxonomy visualize \
 --json output/cora/taxonomy_final.json \
 --output output/cora/taxonomy_graph \
 --format pdf
```

---

## Configuration Highlights

**Model:**
- `gnn_type`: GNN architecture (GCN, GAT, SAGE, GIN, TransformerConv)
- `hidden_dim`: Hidden layer dimension (default: 256)
- `output_dim`: Output embedding dimension (default: 128)
- `n_layers`: Number of GNN layers (default: 2)

**Training:**
- `warmup_epochs`: Warmup pretraining epochs (default: 50)
- `total_epochs`: Total training epochs (default: 100)
- `lr`: Learning rate (default: 0.001)
- `alpha`: Weight for TSA loss (default: 0.5)
- `beta`: Weight for TCA loss (default: 0.3)

**Taxonomy:**
- `max_depth`: Maximum tree depth (default: 3)
- `min_docs`: Minimum documents for splitting (default: 50)
- `over_cluster_factor`: Initial clustering factor (default: 20)
- `taxonomy_update_interval`: Update every N epochs (default: 10)

**Memory Efficiency:**
- `max_pos_samples`: Max positive pairs per batch (default: 2000000)
- `max_neg_samples`: Max negative pairs per batch (default: 2000000)
- `max_tca_samples`: Max documents for TCA (default: 10000)

---

## Usage Examples

### Example 1: Train on Cora with Custom Parameters

```bash
python train.py \
 --dataset cora \
 --gnn_type GAT \
 --hidden_dim 512 \
 --output_dim 256 \
 --warmup_epochs 100 \
 --total_epochs 1000 \
 --alpha 0.7 \
 --beta 0.5 \
 --max_depth 4 \
 --device cuda:0
```

### Example 2: Resume Training from Checkpoint

```bash
python train.py --dataset cora --resume output/cora/model_final.pt
```

### Example 3: Evaluation Only

```bash
python train.py --dataset cora --resume output/cora/model_final.pt --eval_only 1
```

### Example 4: Using Different Text Encoders

```bash
# Use Sentence-BERT
python train.py --dataset cora --text_encoder SentenceBert

# Use E5-Large
python train.py --dataset cora --text_encoder e5-large

# Use LLM (requires more GPU memory)
python train.py --dataset cora --text_encoder Mistral-7B
```

### Example 5: Custom LLM API

```bash
python train.py \
 --dataset cora \
 --api_key YOUR_API_KEY \
 --llm_model gpt-4 \
 --base_url https://api.openai.com/v1
```
