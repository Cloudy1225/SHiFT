"""
train_patent.py

Run SHiFT on the USPTO-CPC patent dataset with hierarchical CPC labels.

Differences from standard train_with_config.py:
  - Loads patent_cpc_dataset.pt + pre-computed LM features
  - Multi-level labels: y_l1 (L1) / y_l2 (L2) / y_l3 (L3)
  - Hierarchical evaluation: Acc@L1, Acc@L2, Acc@L3, hF
  - Temporal train/val/test split (≤2013 / 2014-2015 / ≥2016)

The four training phases (Warmup → Taxonomy → Hierarchy-Aware → Evaluate)
are inherited from HierarchicalTAGLearner unchanged; only data loading and
evaluation are overridden.

Usage (from the SHiFT/ directory):
    python train_patent.py --config configs/patent.yaml
    python train_patent.py --config configs/patent.yaml --override n_runs=3 device=cuda:1
"""

import argparse
import copy
import json
import logging
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.utils import to_undirected

from taxonomy_initialize_patent import LLMClient, CoreSetSampler, TaxonomyBuilder
from taxonomy_update_patent import TaxonomyUpdater
from gnn import GNNEncoder
from model import HierarchicalTAGLearner
from utils import EarlyStopping, get_cur_time, set_random_seed

PATENT_DIR = Path(__file__).parent / "patent"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hierarchical F-score
# ---------------------------------------------------------------------------

def compute_hF(
    pred_l3: torch.Tensor,
    true_l3: torch.Tensor,
    l3_to_ancestors: dict,
) -> dict[str, float]:
    """
    hP = Σ |aug(ŷ) ∩ aug(y)| / Σ |aug(ŷ)|
    hR = Σ |aug(ŷ) ∩ aug(y)| / Σ |aug(y)|
    hF = 2·hP·hR / (hP + hR)
    aug(l3) = {("l1", l1_idx), ("l2", l2_idx), ("l3", l3_idx)}

    Reference: Kiritchenko et al. (2004).
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
    return {"hP": hP, "hR": hR, "hF": hF}


# ---------------------------------------------------------------------------
# Patent-specific subclass
# ---------------------------------------------------------------------------

class PatentHierarchicalLearner(HierarchicalTAGLearner):
    """
    SHiFT learner adapted for USPTO-CPC hierarchical patent classification.

    Overrides:
        __init__                  – patent data loading instead of standard datasets
        _initialize_text_embeddings – load pre-computed LM features
        evaluate                  – hierarchical metrics (L1/L2/L3 Acc + hF)
    """

    def __init__(self, args):
        # Bypass parent __init__ and replicate only what is needed,
        # with patent-specific data loading.
        self.args   = args
        self.device = torch.device(args.device)

        # --- load patent data -------------------------------------------
        logger.info("Loading USPTO-CPC patent dataset...")
        self.data = self._load_patent_data()
        self.data.edge_index = to_undirected(self.data.edge_index.to(self.device))

        # --- text embeddings (load or compute) --------------------------
        self._initialize_text_embeddings()

        # --- GNN encoder ------------------------------------------------
        self.gnn_encoder = GNNEncoder(
            input_dim=self.data.x.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            n_layers=args.n_layers,
            gnn_type=args.gnn_type,
            dropout=args.dropout,
            batch_norm=args.batch_norm,
            residual_conn=args.residual_conn,
            jump_knowledge=args.jump_knowledge,
        ).to(self.device)

        # --- projection head (for TCA) ----------------------------------
        self.projection_head = nn.Sequential(
            nn.Linear(args.output_dim, args.projection_dim),
            nn.ReLU(),
            nn.Linear(args.projection_dim, self.args.text_emb_dim),
        ).to(self.device)

        # --- optimizer --------------------------------------------------
        self.optimizer = torch.optim.Adam(
            list(self.gnn_encoder.parameters())
            + list(self.projection_head.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # --- taxonomy components (initialized lazily) -------------------
        self.taxonomy_root       = None
        self.llm_client          = None
        self.sampler             = None
        self.taxonomy_builder    = None
        self.taxonomy_updater    = None
        self.previous_embeddings = None

        # --- stats / cache ----------------------------------------------
        self.current_epoch  = 0
        self.best_loss      = float("inf")
        self._leaf_assignments_cache     = None
        self._doc_paths_cache            = None
        self._cophenetic_distance_cache  = None
        self._taxonomy_version           = None

        logger.info(
            f"Patent dataset: {self.data.num_nodes} nodes, "
            f"{self.data.edge_index.shape[1]} edges  |  "
            f"L1={self.data.num_l1_classes}  "
            f"L2={self.data.num_l2_classes}  "
            f"L3={self.data.num_l3_classes}  |  "
            f"Train={self.data.train_mask.sum().item()}  "
            f"Val={self.data.val_mask.sum().item()}  "
            f"Test={self.data.test_mask.sum().item()}"
        )

    def _initialize_taxonomy_components(self):
        """Initialize LLM client and taxonomy builder"""
        logger.info("Initializing taxonomy components...")

        self.llm_client = LLMClient(
            api_key=self.args.api_key,
            model=self.args.llm_model,
            base_url=self.args.base_url,
            temperature=self.args.temperature,
            cache_dir=str(Path(self.args.cache_dir) / self.args.dataset)
        )

        self.sampler = CoreSetSampler(
            method=self.args.sampling_method,
            top_k=self.args.top_k
        )

        self.taxonomy_builder = TaxonomyBuilder(
            llm_client=self.llm_client,
            sampler=self.sampler,
            max_depth=self.args.max_depth,
            min_docs=self.args.min_docs,
            over_cluster_factor=self.args.over_cluster_factor,
            llm_max_retries=self.args.llm_max_retries
        )

        self.taxonomy_updater = TaxonomyUpdater(
            llm_client=self.llm_client,
            sampler=self.sampler,
            cosine_threshold=self.args.cosine_threshold,
            merge_threshold=self.args.merge_threshold,
            variance_increase_ratio=self.args.variance_ratio,
            size_increase_ratio=self.args.size_ratio,
            turnover_threshold=self.args.turnover_threshold,
            min_cluster_size=self.args.min_cluster_size,
            max_depth=self.args.max_depth,
            min_docs_for_split=self.args.min_docs,
            llm_max_retries=self.args.llm_max_retries
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_patent_data(self):
        """Load patent_cpc_dataset.pt and wrap in a SimpleNamespace."""
        dataset_path = PATENT_DIR / "patent_cpc_dataset.pt"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                "Run:  python patent/build_patent_cpc_dataset.py"
            )
        raw = torch.load(dataset_path, weights_only=False)

        data = types.SimpleNamespace(
            # graph
            edge_index    = raw["edge_index"],
            # placeholder features (overwritten in _initialize_text_embeddings)
            x             = raw["x"],
            # labels
            y             = raw["y_l3"],    # for compatibility with parent methods
            y_l1          = raw["y_l1"],
            y_l2          = raw["y_l2"],
            y_l3          = raw["y_l3"],
            # masks
            train_mask    = raw["train_mask"],
            val_mask      = raw["val_mask"],
            test_mask     = raw["test_mask"],
            # hierarchy
            l3_to_ancestors   = raw["l3_to_ancestors"],
            num_l1_classes    = raw["num_l1_classes"],
            num_l2_classes    = raw["num_l2_classes"],
            num_l3_classes    = raw["num_l3_classes"],
            # texts for taxonomy construction
            raw_texts     = [
                f"{title} {abstract}".strip() or "Empty text"
                for title, abstract in raw["texts"]
            ],
            num_nodes     = raw["num_nodes"],
        )
        return data

    # ------------------------------------------------------------------
    # Text embeddings (override parent)
    # ------------------------------------------------------------------

    def _initialize_text_embeddings(self):
        """
        Load pre-computed LM features from patent/{encoder}_patent_features.pt.
        Falls back to computing embeddings on the fly if the file is missing.
        """
        encoder_name = self.args.text_encoder
        feat_path    = PATENT_DIR / f"{encoder_name}_patent_features.pt"

        if feat_path.exists():
            logger.info(f"Loading pre-computed {encoder_name} features from {feat_path}")
            text_emb = torch.load(feat_path, weights_only=False, map_location="cpu")
        else:
            logger.warning(
                f"Pre-computed features not found: {feat_path}\n"
                f"Computing on the fly — run encode_patent_features.py to cache them."
            )
            from lm import TextEncoder
            encoder_type = (
                "LM" if encoder_name in ["MiniLM", "SentenceBert", "e5-large", "roberta"]
                else "LLM"
            )
            text_encoder = TextEncoder(encoder_name, encoder_type, self.device)
            text_encoder.model.eval()
            embs, batch_size = [], 32
            with torch.no_grad():
                for i in range(0, len(self.data.raw_texts), batch_size):
                    batch = self.data.raw_texts[i : i + batch_size]
                    embs.append(
                        text_encoder(batch, pooling="mean", max_length=512).cpu()
                    )
            text_emb = torch.cat(embs, dim=0)
            torch.save(text_emb, feat_path)
            logger.info(f"Saved features to {feat_path}")

        self.data.x         = text_emb.float().to(self.device)
        self.args.text_emb_dim = text_emb.shape[1]
        logger.info(f"Node feature shape: {text_emb.shape}")

    # ------------------------------------------------------------------
    # Hierarchical evaluation (override parent)
    # ------------------------------------------------------------------

    def evaluate(self, finetune: bool = True):
        """
        Evaluate GNN representations on hierarchical patent classification.
        Reports Acc@L1, Acc@L2, Acc@L3, hP, hR, hF.
        """
        logger.info("=" * 60)
        logger.info("Hierarchical Patent Classification Evaluation")
        mode = "Fine-tuning GNN" if finetune else "Frozen Features + Linear"
        logger.info(f"Mode: {mode}  |  Runs: {self.args.n_runs}")
        logger.info("=" * 60)

        n_runs = getattr(self.args, "n_runs", 5)
        if finetune:
            results = self._evaluate_hierarchical_finetune(n_runs=n_runs)
        else:
            self.gnn_encoder.eval()
            with torch.no_grad():
                emb = self.gnn_encoder(self.data.x, self.data.edge_index)
            results = self._evaluate_hierarchical_frozen(emb, n_runs=n_runs)

        return results

    # ------------------------------------------------------------------
    # Frozen linear probe
    # ------------------------------------------------------------------

    def _evaluate_hierarchical_frozen(
        self, embeddings: torch.Tensor, n_runs: int = 5
    ) -> dict:
        """Linear probe on frozen GNN embeddings at all three CPC levels."""
        emb_dim  = embeddings.shape[1]
        n_l1     = self.data.num_l1_classes
        n_l2     = self.data.num_l2_classes
        n_l3     = self.data.num_l3_classes
        device   = self.device
        l3_anc   = self.data.l3_to_ancestors

        X           = embeddings.detach()
        train_mask  = self.data.train_mask.to(device)
        val_mask    = self.data.val_mask.to(device)
        test_mask   = self.data.test_mask.to(device)
        y_l1        = self.data.y_l1.to(device)
        y_l2        = self.data.y_l2.to(device)
        y_l3        = self.data.y_l3.to(device)

        run_metrics = {m: [] for m in ["acc_l1", "acc_l2", "acc_l3", "hF"]}

        for run in range(n_runs):
            # Three independent linear heads
            heads = nn.ModuleList([
                nn.Linear(emb_dim, n_l1),
                nn.Linear(emb_dim, n_l2),
                nn.Linear(emb_dim, n_l3),
            ]).to(device)

            opt = torch.optim.Adam(heads.parameters(), lr=0.01, weight_decay=5e-4)

            best_val_loss = float("inf")
            best_state = None
            patience_cnt = 0

            for epoch in range(500):
                heads.train()
                opt.zero_grad()
                logits = [h(X) for h in heads]
                loss = (
                    F.cross_entropy(logits[0][train_mask], y_l1[train_mask])
                    + F.cross_entropy(logits[1][train_mask], y_l2[train_mask])
                    + F.cross_entropy(logits[2][train_mask], y_l3[train_mask])
                )
                loss.backward()
                opt.step()

                heads.eval()
                with torch.no_grad():
                    val_logits = [h(X) for h in heads]
                    val_loss = (
                        F.cross_entropy(val_logits[0][val_mask], y_l1[val_mask])
                        + F.cross_entropy(val_logits[1][val_mask], y_l2[val_mask])
                        + F.cross_entropy(val_logits[2][val_mask], y_l3[val_mask])
                    ).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = copy.deepcopy(heads.state_dict())
                    patience_cnt  = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= 50:
                        break

            heads.load_state_dict(best_state)
            heads.eval()
            with torch.no_grad():
                test_logits = [h(X) for h in heads]

            m = test_mask
            acc_l1 = (test_logits[0].argmax(1)[m] == y_l1[m]).float().mean().item()
            acc_l2 = (test_logits[1].argmax(1)[m] == y_l2[m]).float().mean().item()
            acc_l3 = (test_logits[2].argmax(1)[m] == y_l3[m]).float().mean().item()
            hf     = compute_hF(test_logits[2].argmax(1)[m], y_l3[m], l3_anc)

            run_metrics["acc_l1"].append(acc_l1)
            run_metrics["acc_l2"].append(acc_l2)
            run_metrics["acc_l3"].append(acc_l3)
            run_metrics["hF"].append(hf["hF"])

            logger.info(
                f"  [frozen run {run+1}/{n_runs}] "
                f"L1={acc_l1:.4f}  L2={acc_l2:.4f}  "
                f"L3={acc_l3:.4f}  hF={hf['hF']:.4f}"
            )

        return self._aggregate_and_log(run_metrics, label="Frozen")

    # ------------------------------------------------------------------
    # Full fine-tune probe
    # ------------------------------------------------------------------

    def _evaluate_hierarchical_finetune(self, n_runs: int = 5) -> dict:
        """Fine-tune GNN + three linear heads jointly on hierarchical labels."""
        orig_gnn_state = copy.deepcopy(self.gnn_encoder.state_dict())
        device   = self.device
        n_l1     = self.data.num_l1_classes
        n_l2     = self.data.num_l2_classes
        n_l3     = self.data.num_l3_classes
        l3_anc   = self.data.l3_to_ancestors

        train_mask = self.data.train_mask.to(device)
        val_mask   = self.data.val_mask.to(device)
        test_mask  = self.data.test_mask.to(device)
        y_l1       = self.data.y_l1.to(device)
        y_l2       = self.data.y_l2.to(device)
        y_l3       = self.data.y_l3.to(device)

        finetune_lr = getattr(self.args, "finetune_lr", 1e-3)
        finetune_wd = getattr(self.args, "finetune_weight_decay", 5e-4)
        patience    = getattr(self.args, "finetune_patience", 10)
        ft_epochs   = getattr(self.args, "finetune_epochs", 100)

        run_metrics = {m: [] for m in ["acc_l1", "acc_l2", "acc_l3", "hF"]}

        for run in range(n_runs):
            self.gnn_encoder.load_state_dict(orig_gnn_state)
            heads = nn.ModuleList([
                nn.Linear(self.args.output_dim, n_l1),
                nn.Linear(self.args.output_dim, n_l2),
                nn.Linear(self.args.output_dim, n_l3),
            ]).to(device)

            opt = torch.optim.Adam(
                list(self.gnn_encoder.parameters()) + list(heads.parameters()),
                lr=finetune_lr, weight_decay=finetune_wd,
            )

            best_val_loss = float("inf")
            best_gnn_state   = None
            best_heads_state = None
            patience_cnt     = 0

            for epoch in range(ft_epochs):
                self.gnn_encoder.train()
                heads.train()
                opt.zero_grad()

                z      = self.gnn_encoder(self.data.x, self.data.edge_index)
                logits = [h(z) for h in heads]
                loss   = (
                    F.cross_entropy(logits[0][train_mask], y_l1[train_mask])
                    + F.cross_entropy(logits[1][train_mask], y_l2[train_mask])
                    + F.cross_entropy(logits[2][train_mask], y_l3[train_mask])
                )
                loss.backward()
                opt.step()

                self.gnn_encoder.eval()
                heads.eval()
                with torch.no_grad():
                    z_val      = self.gnn_encoder(self.data.x, self.data.edge_index)
                    val_logits = [h(z_val) for h in heads]
                    val_loss   = (
                        F.cross_entropy(val_logits[0][val_mask], y_l1[val_mask])
                        + F.cross_entropy(val_logits[1][val_mask], y_l2[val_mask])
                        + F.cross_entropy(val_logits[2][val_mask], y_l3[val_mask])
                    ).item()

                if val_loss < best_val_loss:
                    best_val_loss   = val_loss
                    best_gnn_state  = copy.deepcopy(self.gnn_encoder.state_dict())
                    best_heads_state = copy.deepcopy(heads.state_dict())
                    patience_cnt    = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break

            self.gnn_encoder.load_state_dict(best_gnn_state)
            heads.load_state_dict(best_heads_state)

            self.gnn_encoder.eval()
            heads.eval()
            with torch.no_grad():
                z_test     = self.gnn_encoder(self.data.x, self.data.edge_index)
                test_logits = [h(z_test) for h in heads]

            m      = test_mask
            acc_l1 = (test_logits[0].argmax(1)[m] == y_l1[m]).float().mean().item()
            acc_l2 = (test_logits[1].argmax(1)[m] == y_l2[m]).float().mean().item()
            acc_l3 = (test_logits[2].argmax(1)[m] == y_l3[m]).float().mean().item()
            hf     = compute_hF(test_logits[2].argmax(1)[m], y_l3[m], l3_anc)

            run_metrics["acc_l1"].append(acc_l1)
            run_metrics["acc_l2"].append(acc_l2)
            run_metrics["acc_l3"].append(acc_l3)
            run_metrics["hF"].append(hf["hF"])

            logger.info(
                f"  [finetune run {run+1}/{n_runs}] "
                f"L1={acc_l1:.4f}  L2={acc_l2:.4f}  "
                f"L3={acc_l3:.4f}  hF={hf['hF']:.4f}"
            )

        # Restore original GNN weights so subsequent taxonomy calls are valid
        self.gnn_encoder.load_state_dict(orig_gnn_state)
        self.gnn_encoder.eval()

        return self._aggregate_and_log(run_metrics, label="Finetune")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _aggregate_and_log(self, run_metrics: dict, label: str) -> dict:
        results = {}
        for metric, values in run_metrics.items():
            arr = np.array(values)
            results[f"{metric}_mean"] = float(arr.mean())
            results[f"{metric}_std"]  = float(arr.std())

        logger.info(f"\n  [{label}] Classification Summary (test set):")
        for m in ["acc_l1", "acc_l2", "acc_l3", "hF"]:
            logger.info(
                f"    {m:8s}: "
                f"{results[f'{m}_mean']:.4f} ± {results[f'{m}_std']:.4f}"
            )
        return results


# ---------------------------------------------------------------------------
# Logging setup (same pattern as train_with_config.py)
# ---------------------------------------------------------------------------

def setup_logging(tag: str = "patent") -> str:
    os.makedirs("logs", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{tag}_{ts}.log"

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in [logging.FileHandler(log_file, mode="w"), logging.StreamHandler()]:
        handler.setFormatter(fmt)
        logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    return log_file


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def config_to_args(config: dict):
    return argparse.Namespace(**config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run SHiFT on the USPTO-CPC patent dataset."
    )
    parser.add_argument(
        "--config", type=str,
        default="./configs/patent.yaml",
        help="Path to YAML config file (default: configs/patent.yaml)",
    )
    parser.add_argument(
        "--override", type=str, nargs="+",
        help="Override config values, e.g. --override lr=0.01 n_runs=3",
    )
    cmd_args = parser.parse_args()

    config = load_config(cmd_args.config)

    if cmd_args.override:
        for item in cmd_args.override:
            key, value = item.split("=", 1)
            try:
                value = eval(value)
            except Exception:
                pass
            config[key] = value

    args = config_to_args(config)

    log_file = setup_logging(tag=f"patent_{args.gnn_type}")
    logger.info(f"Log → {log_file}")

    set_random_seed(args.random_seed)

    logger.info("=" * 60)
    logger.info("SHiFT — USPTO-CPC Patent Hierarchical Classification")
    logger.info("=" * 60)
    logger.info(f"Start: {get_cur_time()}")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # ---- Phase 1: Warmup -----------------------------------------------
    learner = PatentHierarchicalLearner(args)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Warmup Pretraining")
    logger.info("=" * 60)
    learner.warmup_pretrain()
    learner.evaluate(finetune=args.finetune)

    # ---- Phase 2: Initial taxonomy -------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Initial Taxonomy Construction")
    logger.info("=" * 60)
    learner.build_initial_taxonomy()

    # ---- Phase 3: Hierarchy-aware training -----------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Hierarchy-Aware Training")
    logger.info("=" * 60)
    learner.hierarchy_aware_training()

    # ---- Phase 4: Final evaluation -------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Final Evaluation")
    logger.info("=" * 60)
    final_results = learner.evaluate(finetune=args.finetune)

    # Save final results
    out_dir = Path(args.output_dir) / "patent"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "hierarchical_results.json"
    out_json.write_text(json.dumps(final_results, indent=2))
    logger.info(f"Results saved → {out_json}")

    logger.info("=" * 60)
    logger.info(f"End: {get_cur_time()}")
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
