# USPTO-CPC Hierarchical Patent Classification Dataset

A text-attributed citation graph dataset built from [USPTO20000](https://github.com/opendata-ai/tr) with ground-truth hierarchical labels from the [Cooperative Patent Classification (CPC)](https://www.cooperativepatentclassification.org/) system. Designed to benchmark GNNs' ability to learn and exploit semantic hierarchies.

---

## Motivation

This dataset was constructed as prior TAG benchmarks **lack verifiable ground-truth hierarchies**. CPC is a 5-level taxonomy jointly maintained by the USPTO and EPO, manually assigned by patent examiners — making it an authoritative, externally verifiable hierarchical label structure. The patent citation graph further provides a unique property: citation edges carry quantifiably different amounts of hierarchical signal at each CPC level (93.9 % same-section rate at L1 vs. 47.1 % at L5), demonstrating that the graph structure itself encodes multi-granular semantic proximity.

---

## Dataset Statistics

| Property | Value |
|---|---|
| Nodes (patents) | 10,090 |
| Edges (undirected citations) | 19,722 |
| Node features | 1,024-dim RoBERTa embeddings (title + abstract) |
| **L1** classes (CPC Section) | 7 |
| **L2** classes (CPC Class) | 55 |
| **L3** classes (CPC Subclass) | 162 |
| Train split | 6,956 patents (grant year ≤ 2013) |
| Val split | 1,549 patents (grant year 2014–2015) |
| Test split | 1,585 patents (grant year ≥ 2016) |

The split is **temporal**: models are trained on older patents and evaluated on newer ones, matching the realistic patent retrieval/classification scenario and preventing any future-information leakage.

### CPC Hierarchy

CPC codes have a strict 5-level tree structure. We use the first three levels:

```
L1  Section     (e.g., H)           —  7 classes
L2  Class       (e.g., H04)         — 55 classes
L3  Subclass    (e.g., H04L)        — 162 classes   ← finest level used
L4  Main Group  (e.g., H04L63)      —  not used
L5  Subgroup    (e.g., H04L63/0281) —  not used
```

Each patent is assigned the **primary inventive** CPC code, yielding a unique root-to-leaf path `L1 > L2 > L3` per node. L3 classes with fewer than 10 samples are discarded.

### Graph-Level Evidence of Hierarchical Structure

The following table shows the fraction of citation edges where citing and cited patents share the same CPC label at each level. This gradient confirms that the citation graph encodes hierarchical semantic proximity:

| CPC Level | Same-label edge rate |
|---|---|
| L1 (Section) | 93.9 % |
| L2 (Class) | 89.7 % |
| L3 (Subclass) | 84.3 % |
| L4 (Main Group) | 75.4 % |
| L5 (Subgroup) | 47.1 % |

---

## File Structure

```
patent/
├── README.md
├── USPTO20000.txt                  # Raw USPTO20000 input file
├── patent_cpc_long.csv             # CPC labels per patent (from Google Patents BigQuery)
├── patent_cpc_dataset.pt           # Processed PyG-compatible dataset (generated)
├── patent_cpc_dataset_meta.json    # Dataset statistics (generated)
│
├── build_uspto20000_graph.py       # Step 0: build citation graph from USPTO20000.txt
├── build_patent_cpc_dataset.py     # Step 1: construct hierarchical labeled dataset
├── encode_patent_features.py       # Step 2: encode title+abstract into LM embeddings
├── train_patent_gnn.py             # Step 3: train and evaluate MLP/GCN/SAGE/GAT
│
├── uspto20000_graph/               # Citation graph artifacts (from Step 0)
│   ├── nodes.tsv
│   ├── edges.tsv
│   ├── patent_ids.csv
│   ├── citation_map.json
│   ├── stats.json
│   └── graph_data.pt
│
└── runs/                           # Training outputs (from Step 3, generated)
    ├── MLP_roberta/
    │   ├── ckpt_run{i}.pt
    │   └── results_5runs.json
    ├── GCN_roberta/
    ├── SAGE_roberta/
    └── GAT_roberta/
```

---

## Reproducing Results

### Prerequisites

```bash
pip install torch torch-geometric transformers sentence-transformers tqdm
```

### Step 0 — Build Citation Graph

```bash
python patent/build_uspto20000_graph.py USPTO20000.txt --output-dir patent/uspto20000_graph
```

Parses the raw USPTO20000 file and writes `nodes.tsv`, `edges.tsv`, and `patent_ids.csv`. Reissue patents (RE-prefix) are automatically excluded.

### Step 1 — Build Hierarchical Dataset

```bash
python patent/build_patent_cpc_dataset.py
```

Joins `nodes.tsv` / `edges.tsv` with CPC labels from `patent_cpc_long.csv`. Applies temporal split and saves `patent_cpc_dataset.pt`.

### Step 2 — Encode Node Features

```bash
python patent/encode_patent_features.py --encoder roberta --batch_size 32
```

Encodes `"{title} {abstract}"` for every patent using `sentence-transformers/all-roberta-large-v1` (mean pooling). Saves `roberta_patent_features.pt` (10,090 × 1,024). Supported encoders: `MiniLM`, `SentenceBert`, `e5-large`, `roberta`.

### Step 3 — Train and Evaluate

```bash
# Single model
python patent/train_patent_gnn.py --model GCN --encoder roberta --n_runs 5

# All models (MLP, GCN, SAGE, GAT) with summary table
python patent/train_patent_gnn.py --run_all --encoder roberta --n_runs 5
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `GCN` | `MLP`, `GCN`, `SAGE`, `GAT`, `GIN` |
| `--run_all` | — | Run MLP + GCN + SAGE + GAT sequentially |
| `--encoder` | `roberta` | LM used for node features |
| `--n_runs` | `5` | Number of independent runs (mean ± std) |
| `--seed` | `42` | Base seed; run *i* uses `seed + i` |
| `--n_layers` | `2` | GNN/MLP depth |
| `--hidden_dim` | `256` | Hidden dimension |
| `--epochs` | `500` | Max training epochs |
| `--patience` | `50` | Early stopping patience |

---

## Results

Test-set performance (mean ± std over 5 independent runs, RoBERTa features, temporal split).

| Model | L1 Acc | L2 Acc | L3 Acc | hF |
|---|---|---|---|---|
| MLP | 0.8132 ± 0.0049 | 0.7365 ± 0.0065 | 0.6432 ± 0.0058 | 0.7345 ± 0.0042 |
| GCN | **0.8384 ± 0.0018** | **0.7852 ± 0.0033** | **0.7091 ± 0.0052** | **0.7785 ± 0.0039** |
| SAGE | 0.8413 ± 0.0057 | 0.7759 ± 0.0036 | 0.6972 ± 0.0043 | 0.7735 ± 0.0035 |
| GAT | 0.8323 ± 0.0058 | 0.7763 ± 0.0039 | 0.7008 ± 0.0048 | 0.7691 ± 0.0051 |

**hF** is the hierarchical F-score (Kiritchenko et al., 2004), computed from L3 predictions by augmenting each label with its L2 and L1 ancestors:
$$
hF = \frac{2 \cdot hP \cdot hR}{hP + hR}, \quad hP = \frac{\sum_i |\hat{Y}_i \cap \text{aug}(Y_i)|}{\sum_i |\text{aug}(\hat{Y}_i)|}, \quad hR = \frac{\sum_i |\hat{Y}_i \cap \text{aug}(Y_i)|}{\sum_i |\text{aug}(Y_i)|}
$$
Key observations:
- All three GNN variants outperform the text-only MLP across all levels and the hF metric, confirming that citation graph structure carries hierarchical semantic signal beyond what text alone provides.
- The GNN advantage is most pronounced at L3 (finest granularity): GCN improves over MLP by **+6.6 pp** in L3 accuracy and **+4.4 pp** in hF, compared to **+2.5 pp** at L1. This is consistent with the empirical observation that coarser CPC levels are already well-encoded in text, while finer-grained distinctions benefit from graph-based neighborhood aggregation.
- The hF metric captures hierarchical consistency beyond accuracy: a model may achieve moderate L3 accuracy but still violate parent–child consistency, leading to lower hF. GNNs trained with joint multi-level loss exhibit higher hierarchical consistency than the MLP baseline.

---

## CPC Label Source

CPC labels were queried from the [Google Patents Public Data](https://console.cloud.google.com/marketplace/product/google_patents_public_data/google-patents-public-data) BigQuery dataset (`patents-public-data.patents.publications`). Only the **primary inventive** CPC code per patent is used to ensure a unique root-to-leaf label path. The `cpc_path` column was backfilled by parsing the CPC code string directly (the `tree` field in the BigQuery table is sparsely populated).

```sql
WITH requested_ids AS (
  SELECT DISTINCT
    REGEXP_REPLACE(
      CAST(patent_id AS STRING),
      r'[^0-9]',
      ''
    ) AS patent_id
  FROM `patent-cpc-query.patent_data.patent_ids`
),

matched_patents AS (
  SELECT
    ids.patent_id,
    p.publication_number,
    p.kind_code,
    p.grant_date,
    p.family_id,
    p.cpc,

    ROW_NUMBER() OVER (
      PARTITION BY ids.patent_id
      ORDER BY
        CASE
          WHEN p.kind_code IN ('B1', 'B2') THEN 0
          ELSE 1
        END,
        p.grant_date DESC,
        p.publication_number
    ) AS match_rank

  FROM requested_ids AS ids

  JOIN `patents-public-data.patents.publications` AS p
    ON p.country_code = 'US'
   AND REGEXP_EXTRACT(
         REPLACE(p.publication_number, '-', ''),
         r'^US0*([0-9]+)[A-Z]'
       ) = ids.patent_id

  WHERE
    p.grant_date IS NOT NULL
    AND p.grant_date > 0
),

selected_patents AS (
  SELECT *
  FROM matched_patents
  WHERE match_rank = 1
)

SELECT DISTINCT
  p.patent_id,
  p.publication_number,
  p.kind_code,
  p.grant_date,
  CAST(p.family_id AS STRING) AS family_id,

  REGEXP_REPLACE(c.code, r'\s+', '') AS cpc_code,

  c.inventive AS is_inventive,
  c.first AS is_primary,

  ARRAY_TO_STRING(c.tree, '>') AS cpc_path

FROM selected_patents AS p
CROSS JOIN UNNEST(p.cpc) AS c

WHERE c.code IS NOT NULL

ORDER BY
  patent_id,
  is_primary DESC,
  is_inventive DESC,
  cpc_code;
```

