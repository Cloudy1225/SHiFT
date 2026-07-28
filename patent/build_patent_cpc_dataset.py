#!/usr/bin/env python3
"""
Build USPTO-CPC hierarchical graph dataset for node classification.

Inputs (relative to this file's directory):
  uspto20000_graph/nodes.tsv   — patent_id, title, abstract
  uspto20000_graph/edges.tsv   — src/dst patent citation edges
  patent_cpc_long.csv          — CPC labels (primary + inventive)

Outputs:
  patent_cpc_dataset.pt        — dataset dict (PyG-compatible)
  patent_cpc_dataset_meta.json — human-readable stats
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PATENT_DIR = Path(__file__).parent
GRAPH_DIR  = PATENT_DIR / "uspto20000_graph"

MIN_SAMPLES_L3 = 10  # drop L3 classes with fewer samples

# Temporal split boundaries (grant year)
#   train : year <= TRAIN_YEAR_END   (~61 % of patents)
#   val   : TRAIN_YEAR_END < year <= VAL_YEAR_END  (~18 %)
#   test  : year > VAL_YEAR_END      (~21 %)
TRAIN_YEAR_END = 2013
VAL_YEAR_END   = 2015


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cpc_to_l1_l2_l3(code: str) -> tuple[str, str, str] | None:
    """Parse a CPC code into (L1-section, L2-class, L3-subclass), or None."""
    m = re.match(r'^([A-Z])(\d{2})([A-Z])\d+(/.*)?$', code.strip())
    if not m:
        return None
    s, cl, sc = m.group(1), m.group(2), m.group(3)
    return s, s + cl, s + cl + sc


def temporal_split(
    grant_years: np.ndarray,
    train_year_end: int = TRAIN_YEAR_END,
    val_year_end:   int = VAL_YEAR_END,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return boolean train/val/test masks based on grant year.

        train : grant_year <= train_year_end
        val   : train_year_end < grant_year <= val_year_end
        test  : grant_year > val_year_end

    Patents with unknown year (year == 0) fall into train to avoid leakage.
    """
    train_mask = (grant_years <= train_year_end) | (grant_years == 0)
    val_mask   = (grant_years > train_year_end) & (grant_years <= val_year_end)
    test_mask  = (grant_years > val_year_end)
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset() -> dict:
    # ------------------------------------------------------------------
    # 1. Load primary+inventive CPC per patent
    # ------------------------------------------------------------------
    patent_cpc: dict[str, tuple[str, str, str, int]] = {}
    # patent_id -> (l1, l2, l3, grant_year)

    with open(PATENT_DIR / "patent_cpc_long.csv") as f:
        for row in csv.DictReader(f):
            if row["is_primary"] != "true" or row["is_inventive"] != "true":
                continue
            pid = row["patent_id"].strip()
            if pid in patent_cpc:
                continue  # keep only the first (highest-rank) primary code
            levels = cpc_to_l1_l2_l3(row["cpc_code"])
            if levels is None:
                continue
            try:
                grant_year = int(str(row["grant_date"]).strip()[:4])
            except (ValueError, TypeError):
                grant_year = 0
            patent_cpc[pid] = (*levels, grant_year)

    print(f"Patents with primary+inventive CPC: {len(patent_cpc)}")

    # ------------------------------------------------------------------
    # 2. Filter: keep only L3 classes with >= MIN_SAMPLES_L3 patents
    # ------------------------------------------------------------------
    l3_counts = Counter(v[2] for v in patent_cpc.values())
    valid_l3  = {lbl for lbl, cnt in l3_counts.items() if cnt >= MIN_SAMPLES_L3}
    patent_cpc = {pid: v for pid, v in patent_cpc.items() if v[2] in valid_l3}
    print(f"After L3 ≥{MIN_SAMPLES_L3} filter: {len(patent_cpc)} patents")

    # ------------------------------------------------------------------
    # 3. Load node text (title + abstract) from nodes.tsv
    # ------------------------------------------------------------------
    node_text: dict[str, tuple[str, str]] = {}
    with open(GRAPH_DIR / "nodes.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            node_text[row["patent_id"].strip()] = (
                row["title"].strip(),
                row["abstract"].strip(),
            )

    # Intersect: only patents present in both CPC and text
    valid_pids = sorted(
        set(patent_cpc) & set(node_text),
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )
    print(f"Patents with both CPC and text: {len(valid_pids)}")

    # ------------------------------------------------------------------
    # 4. Build deterministic label encoders (alphabetical order)
    # ------------------------------------------------------------------
    def make_encoder(iterable) -> dict[str, int]:
        return {lbl: i for i, lbl in enumerate(sorted(set(iterable)))}

    l1_enc = make_encoder(patent_cpc[p][0] for p in valid_pids)
    l2_enc = make_encoder(patent_cpc[p][1] for p in valid_pids)
    l3_enc = make_encoder(patent_cpc[p][2] for p in valid_pids)

    pid_to_idx = {pid: i for i, pid in enumerate(valid_pids)}
    n = len(valid_pids)

    y_l1 = torch.tensor([l1_enc[patent_cpc[p][0]] for p in valid_pids], dtype=torch.long)
    y_l2 = torch.tensor([l2_enc[patent_cpc[p][1]] for p in valid_pids], dtype=torch.long)
    y_l3 = torch.tensor([l3_enc[patent_cpc[p][2]] for p in valid_pids], dtype=torch.long)
    grant_years = torch.tensor([patent_cpc[p][3] for p in valid_pids], dtype=torch.long)

    cpc_paths = [
        f"{patent_cpc[p][0]}>{patent_cpc[p][1]}>{patent_cpc[p][2]}"
        for p in valid_pids
    ]

    # ------------------------------------------------------------------
    # 5. Build ancestor map:  l3_idx -> (l2_idx, l1_idx)
    #    Used by hF computation in the training script.
    # ------------------------------------------------------------------
    l3_to_ancestors: dict[int, tuple[int, int]] = {}
    for p in valid_pids:
        l1, l2, l3 = patent_cpc[p][:3]
        l3_idx = l3_enc[l3]
        if l3_idx not in l3_to_ancestors:
            l3_to_ancestors[l3_idx] = (l2_enc[l2], l1_enc[l1])

    # ------------------------------------------------------------------
    # 6. Build undirected edge_index (re-indexed to valid_pids only)
    # ------------------------------------------------------------------
    src_list, dst_list = [], []
    with open(GRAPH_DIR / "edges.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s = row["src_patent_id"].strip()
            d = row["dst_patent_id"].strip()
            if s in pid_to_idx and d in pid_to_idx:
                src_list.append(pid_to_idx[s])
                dst_list.append(pid_to_idx[d])

    # Make undirected and remove duplicate edges
    edge_index = torch.tensor(
        [src_list + dst_list, dst_list + src_list],
        dtype=torch.long,
    )
    edge_index = torch.unique(edge_index, dim=1)
    print(f"Undirected edges: {edge_index.shape[1]}")

    # ------------------------------------------------------------------
    # 7. Temporal train / val / test split
    #    train : year <= 2013  |  val : 2014-2015  |  test : >= 2016
    # ------------------------------------------------------------------
    train_mask, val_mask, test_mask = temporal_split(grant_years.numpy())
    print(
        f"Temporal split — "
        f"train (≤{TRAIN_YEAR_END}): {train_mask.sum()}  "
        f"val ({TRAIN_YEAR_END+1}–{VAL_YEAR_END}): {val_mask.sum()}  "
        f"test (≥{VAL_YEAR_END+1}): {test_mask.sum()}"
    )

    # ------------------------------------------------------------------
    # 8. Pack and save
    # ------------------------------------------------------------------
    dataset = {
        # Graph topology
        "edge_index":   edge_index,
        # Node features: placeholder zeros; replaced by encode_patent_features.py
        "x":            torch.zeros(n, 1),
        # Hierarchical labels
        "y_l1":         y_l1,
        "y_l2":         y_l2,
        "y_l3":         y_l3,
        # Masks
        "train_mask":   torch.from_numpy(train_mask),
        "val_mask":     torch.from_numpy(val_mask),
        "test_mask":    torch.from_numpy(test_mask),
        # Metadata
        "patent_ids":       valid_pids,
        "texts":            [node_text[p] for p in valid_pids],
        "grant_years":      grant_years,
        "cpc_paths":        cpc_paths,
        "label_maps":       {"l1": l1_enc, "l2": l2_enc, "l3": l3_enc},
        "l3_to_ancestors":  l3_to_ancestors,
        "num_nodes":        n,
        "num_l1_classes":   len(l1_enc),
        "num_l2_classes":   len(l2_enc),
        "num_l3_classes":   len(l3_enc),
    }

    out_pt = PATENT_DIR / "patent_cpc_dataset.pt"
    torch.save(dataset, out_pt)

    meta = {
        "num_nodes":            n,
        "num_edges_undirected": int(edge_index.shape[1]),
        "num_l1_classes":       len(l1_enc),
        "num_l2_classes":       len(l2_enc),
        "num_l3_classes":       len(l3_enc),
        "split": {
            "train": int(train_mask.sum()),
            "val":   int(val_mask.sum()),
            "test":  int(test_mask.sum()),
            "train_years": f"≤{TRAIN_YEAR_END}",
            "val_years":   f"{TRAIN_YEAR_END+1}–{VAL_YEAR_END}",
            "test_years":  f"≥{VAL_YEAR_END+1}",
        },
        "l1_classes": sorted(l1_enc.keys()),
        "l3_min_samples": MIN_SAMPLES_L3,
    }
    (PATENT_DIR / "patent_cpc_dataset_meta.json").write_text(
        json.dumps(meta, indent=2)
    )

    print(f"\nSaved → {out_pt}")
    print(json.dumps(meta, indent=2))
    return dataset


if __name__ == "__main__":
    build_dataset()
