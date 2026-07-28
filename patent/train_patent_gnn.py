#!/usr/bin/env python3
"""
Train MLP / GCN / GraphSAGE / GAT on USPTO-CPC for hierarchical node classification.

Models
------
  MLP  — text-only baseline; ignores graph edges
  GCN  — Graph Convolutional Network
  SAGE — GraphSAGE
  GAT  — Graph Attention Network

Three classification heads share one backbone:
    L1 ( 8 classes)  → CPC Section
    L2 (≈60 classes) → CPC Class
    L3 (≈162 classes)→ CPC Subclass

Training objective:  loss = CE(L1) + CE(L2) + CE(L3)

Evaluation metrics:
    Acc@L1, Acc@L2, Acc@L3  — per-level accuracy
    hF                       — hierarchical F-score (Kiritchenko et al., 2004)

Outputs (per model)
-------------------
  patent/runs/{model}_{encoder}/
      ckpt_run{i}.pt           — best checkpoint for run i
      results_{n}runs.json     — aggregated mean ± std + per-run details

Usage
-----
    python train_patent_gnn.py --model MLP  --encoder roberta
    python train_patent_gnn.py --model GCN  --encoder roberta
    python train_patent_gnn.py --run_all    --encoder roberta  --n_runs 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATENT_DIR = Path(__file__).parent
SHIFT_DIR  = PATENT_DIR.parent
sys.path.insert(0, str(SHIFT_DIR))

from gnn import GNNEncoder                    # noqa: E402
from utils import set_random_seed, EarlyStopping  # noqa: E402


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MultiLevelMLP(nn.Module):
    """
    Text-only baseline.  Ignores graph structure entirely.
    Architecture:  Linear → BN → ReLU → Dropout  (× n_layers)  → 3 heads
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        emb_dim:    int,
        n_l1:       int,
        n_l2:       int,
        n_l3:       int,
        n_layers:   int   = 2,
        dropout:    float = 0.5,
    ):
        super().__init__()
        dims   = [in_dim] + [hidden_dim] * (n_layers - 1) + [emb_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.encoder = nn.Sequential(*layers)
        self.head_l1 = nn.Linear(emb_dim, n_l1)
        self.head_l2 = nn.Linear(emb_dim, n_l2)
        self.head_l3 = nn.Linear(emb_dim, n_l3)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.head_l1(z), self.head_l2(z), self.head_l3(z)


class MultiLevelGNN(nn.Module):
    """
    Shared GNN backbone + three independent classification heads.
    Supports GCN, SAGE, GAT, GIN via the existing GNNEncoder.
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        emb_dim:    int,
        n_l1:       int,
        n_l2:       int,
        n_l3:       int,
        gnn_type:   str   = "GCN",
        n_layers:   int   = 2,
        dropout:    float = 0.5,
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            output_dim=emb_dim,
            n_layers=n_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            batch_norm=1,
            residual_conn=1,
        )
        self.head_l1 = nn.Linear(emb_dim, n_l1)
        self.head_l2 = nn.Linear(emb_dim, n_l2)
        self.head_l3 = nn.Linear(emb_dim, n_l3)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x, edge_index)
        return self.head_l1(z), self.head_l2(z), self.head_l3(z)


def build_model(
    model_name: str, in_dim: int, n_l1: int, n_l2: int, n_l3: int, args: argparse.Namespace
) -> nn.Module:
    kwargs = dict(
        in_dim=in_dim, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim,
        n_l1=n_l1, n_l2=n_l2, n_l3=n_l3,
        n_layers=args.n_layers, dropout=args.dropout,
    )
    if model_name == "MLP":
        return MultiLevelMLP(**kwargs)
    return MultiLevelGNN(gnn_type=model_name, **kwargs)


# ---------------------------------------------------------------------------
# Hierarchical F-score (hF)
# ---------------------------------------------------------------------------

def compute_hF(
    pred_l3:         torch.Tensor,
    true_l3:         torch.Tensor,
    l3_to_ancestors: dict[int, tuple[int, int]],
) -> dict[str, float]:
    """
    hP = Σ |aug(ŷ) ∩ aug(y)| / Σ |aug(ŷ)|
    hR = Σ |aug(ŷ) ∩ aug(y)| / Σ |aug(y)|
    hF = 2·hP·hR / (hP + hR)

    aug(l3_idx) = {("l1", l1_idx), ("l2", l2_idx), ("l3", l3_idx)}

    Reference:
        Kiritchenko et al. (2004). Functional Annotation of Genes Using
        Hierarchical Text Categorization. BioLINK SIG Workshop.
    """
    pred = pred_l3.cpu().numpy()
    true = true_l3.cpu().numpy()
    total_correct = total_pred = total_true = 0

    for p, t in zip(pred, true):
        p_l2, p_l1 = l3_to_ancestors.get(int(p), (int(p), int(p)))
        t_l2, t_l1 = l3_to_ancestors.get(int(t), (int(t), int(t)))
        aug_pred = {("l1", p_l1), ("l2", p_l2), ("l3", int(p))}
        aug_true = {("l1", t_l1), ("l2", t_l2), ("l3", int(t))}
        total_correct += len(aug_pred & aug_true)
        total_pred    += len(aug_pred)
        total_true    += len(aug_true)

    hP = total_correct / total_pred if total_pred else 0.0
    hR = total_correct / total_true if total_true else 0.0
    hF = 2 * hP * hR / (hP + hR) if (hP + hR) else 0.0
    return {"hP": round(hP, 6), "hR": round(hR, 6), "hF": round(hF, 6)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:           nn.Module,
    data:            dict,
    mask:            torch.Tensor,
    l3_to_ancestors: dict,
) -> dict[str, float]:
    model.eval()
    logit_l1, logit_l2, logit_l3 = model(data["x"], data["edge_index"])

    m       = mask
    pred_l1 = logit_l1.argmax(dim=1)[m]
    pred_l2 = logit_l2.argmax(dim=1)[m]
    pred_l3 = logit_l3.argmax(dim=1)[m]
    true_l1 = data["y_l1"][m]
    true_l2 = data["y_l2"][m]
    true_l3 = data["y_l3"][m]

    acc_l1 = (pred_l1 == true_l1).float().mean().item()
    acc_l2 = (pred_l2 == true_l2).float().mean().item()
    acc_l3 = (pred_l3 == true_l3).float().mean().item()
    loss   = (
        F.cross_entropy(logit_l1[m], true_l1)
        + F.cross_entropy(logit_l2[m], true_l2)
        + F.cross_entropy(logit_l3[m], true_l3)
    ).item()

    return {
        "loss":   loss,
        "acc_l1": round(acc_l1, 6),
        "acc_l2": round(acc_l2, 6),
        "acc_l3": round(acc_l3, 6),
        **compute_hF(pred_l3, true_l3, l3_to_ancestors),
    }


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_one(
    args:       argparse.Namespace,
    model_name: str,
    seed:       int,
    run_idx:    int,
    run_dir:    Path,
) -> dict:
    set_random_seed(seed)
    device = torch.device(args.device)

    # -- Load dataset --------------------------------------------------------
    dataset_path = PATENT_DIR / "patent_cpc_dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run:  python build_patent_cpc_dataset.py"
        )
    dataset = torch.load(dataset_path, weights_only=False)

    feat_path = PATENT_DIR / f"{args.encoder}_patent_features.pt"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features not found: {feat_path}\n"
            f"Run:  python encode_patent_features.py --encoder {args.encoder}"
        )
    x = torch.load(feat_path, weights_only=False, map_location="cpu").float()

    data = {
        "x":          x.to(device),
        "edge_index": dataset["edge_index"].to(device),
        "y_l1":       dataset["y_l1"].to(device),
        "y_l2":       dataset["y_l2"].to(device),
        "y_l3":       dataset["y_l3"].to(device),
        "train_mask": dataset["train_mask"].to(device),
        "val_mask":   dataset["val_mask"].to(device),
        "test_mask":  dataset["test_mask"].to(device),
    }

    l3_to_ancestors = dataset["l3_to_ancestors"]
    n_l1, n_l2, n_l3 = (
        dataset["num_l1_classes"],
        dataset["num_l2_classes"],
        dataset["num_l3_classes"],
    )
    in_dim = x.shape[1]

    # -- Model ---------------------------------------------------------------
    model     = build_model(model_name, in_dim, n_l1, n_l2, n_l3, args).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    ckpt_path = str(run_dir / f"ckpt_run{run_idx}.pt")

    print(
        f"[{model_name} run{run_idx}] "
        f"nodes={dataset['num_nodes']}  edges={dataset['edge_index'].shape[1]}  "
        f"feat={in_dim}  params={n_params:,}  seed={seed}"
    )

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    early_stop = EarlyStopping(patience=args.patience, path=ckpt_path)

    # -- Training loop -------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logit_l1, logit_l2, logit_l3 = model(data["x"], data["edge_index"])
        m = data["train_mask"]
        loss = (
            F.cross_entropy(logit_l1[m], data["y_l1"][m])
            + F.cross_entropy(logit_l2[m], data["y_l2"][m])
            + F.cross_entropy(logit_l3[m], data["y_l3"][m])
        )
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            val = evaluate(model, data, data["val_mask"], l3_to_ancestors)
            early_stop(val["loss"], model)
            print(
                f"[{model_name} run{run_idx}] ep {epoch:4d} | "
                f"train={loss.item():.4f} | "
                f"val L1={val['acc_l1']:.4f} "
                f"L2={val['acc_l2']:.4f} "
                f"L3={val['acc_l3']:.4f} "
                f"hF={val['hF']:.4f}"
            )
            if early_stop.early_stop:
                print(f"[{model_name} run{run_idx}] Early stopping at epoch {epoch}.")
                break

    # -- Evaluate best checkpoint --------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    val_metrics  = evaluate(model, data, data["val_mask"],  l3_to_ancestors)
    test_metrics = evaluate(model, data, data["test_mask"], l3_to_ancestors)

    print(
        f"[{model_name} run{run_idx}] "
        f"Val  L1={val_metrics['acc_l1']:.4f} "
        f"L2={val_metrics['acc_l2']:.4f} "
        f"L3={val_metrics['acc_l3']:.4f} "
        f"hF={val_metrics['hF']:.4f}"
    )
    print(
        f"[{model_name} run{run_idx}] "
        f"Test L1={test_metrics['acc_l1']:.4f} "
        f"L2={test_metrics['acc_l2']:.4f} "
        f"L3={test_metrics['acc_l3']:.4f} "
        f"hF={test_metrics['hF']:.4f}"
    )

    return {
        "model":   model_name,
        "encoder": args.encoder,
        "seed":    seed,
        "run_idx": run_idx,
        "val":     val_metrics,
        "test":    test_metrics,
    }


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

METRICS = ["acc_l1", "acc_l2", "acc_l3", "hF"]


def run_multi(args: argparse.Namespace, model_name: str) -> dict:
    """
    Run `args.n_runs` independent training runs (seeds: args.seed … args.seed+n-1).
    Saves checkpoints and aggregated JSON under patent/runs/{model_name}_{encoder}/.
    """
    run_dir = PATENT_DIR / "runs" / f"{model_name}_{args.encoder}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds        = [args.seed + i for i in range(args.n_runs)]
    run_results  = []

    for i, seed in enumerate(seeds):
        print(f"\n{'─'*65}")
        print(f"  {model_name}  run {i+1}/{args.n_runs}  (seed={seed})")
        print(f"{'─'*65}")
        r = train_one(args, model_name, seed=seed, run_idx=i, run_dir=run_dir)
        run_results.append(r)

    # Aggregate mean ± std
    def stats(values: list[float]) -> dict:
        arr = np.array(values)
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    aggregated = {
        "model":   model_name,
        "encoder": args.encoder,
        "n_runs":  args.n_runs,
        "seeds":   seeds,
        "test": {m: stats([r["test"][m] for r in run_results]) for m in METRICS},
        "val":  {m: stats([r["val"][m]  for r in run_results]) for m in METRICS},
        "runs": run_results,
    }

    out_json = run_dir / f"results_{args.n_runs}runs.json"
    out_json.write_text(json.dumps(aggregated, indent=2))
    print(f"\nResults → {out_json}")

    return aggregated


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_agg: list[dict]) -> None:
    """Print mean ± std table across all models."""
    col    = 16
    header = (
        f"{'Model':<10} {'Enc':<10} "
        + "  ".join(f"{m:>{col}}" for m in ["L1 Acc", "L2 Acc", "L3 Acc", "hF"])
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for agg in all_agg:
        t    = agg["test"]
        vals = [
            f"{t['acc_l1']['mean']:.4f}±{t['acc_l1']['std']:.4f}",
            f"{t['acc_l2']['mean']:.4f}±{t['acc_l2']['std']:.4f}",
            f"{t['acc_l3']['mean']:.4f}±{t['acc_l3']['std']:.4f}",
            f"{t['hF']['mean']:.4f}±{t['hF']['std']:.4f}",
        ]
        print(
            f"{agg['model']:<10} {agg['encoder']:<10} "
            + "  ".join(f"{v:>{col}}" for v in vals)
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLP/GNN baselines on USPTO-CPC hierarchical classification."
    )
    parser.add_argument(
        "--model", type=str, default="GCN",
        choices=["MLP", "GCN", "SAGE", "GAT", "GIN"],
        help="Model to train (default: GCN; ignored when --run_all is set)",
    )
    parser.add_argument(
        "--run_all", action="store_true",
        help="Run MLP, GCN, SAGE, GAT sequentially and print summary table",
    )
    parser.add_argument("--encoder",      type=str,   default="roberta",
                        help="LM encoder used for node features (default: roberta)")
    parser.add_argument("--n_layers",     type=int,   default=2)
    parser.add_argument("--hidden_dim",   type=int,   default=256)
    parser.add_argument("--emb_dim",      type=int,   default=256)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--wd",           type=float, default=5e-4)
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--patience",     type=int,   default=50)
    parser.add_argument("--log_interval", type=int,   default=10)
    parser.add_argument("--n_runs",       type=int,   default=5,
                        help="Number of independent runs (default: 5)")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Base seed; run i uses seed+i (default: 42)")
    parser.add_argument(
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    models_to_run = ["MLP", "GCN", "SAGE", "GAT"] if args.run_all else [args.model]
    all_agg       = [run_multi(args, m) for m in models_to_run]

    print_summary(all_agg)
