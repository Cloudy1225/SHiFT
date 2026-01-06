"""
evaluate_taxonomy_as_features.py

Evaluate taxonomy quality by augmenting node features with hierarchical path information
"""

import argparse
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import copy

from dataloader import load_graph_dataset
from gnn import GNNEncoder
from taxonomy import TaxonomyLoader, TaxonomyQuery
from lm import TextEncoder
from utils import set_random_seed, EarlyStopping

logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PathTextExtractor:
    """Extract and format taxonomy path text for each node"""

    def __init__(self, taxonomy_root, include_definitions=True,
                 include_names=True, separator=" [SEP] "):
        """
        Args:
            taxonomy_root: Root of the taxonomy tree
            include_definitions: Whether to include concept definitions
            include_names: Whether to include concept names
            separator: Separator between path elements
        """
        self.taxonomy_root = taxonomy_root
        self.include_definitions = include_definitions
        self.include_names = include_names
        self.separator = separator
        self.query = TaxonomyQuery(taxonomy_root)

        # Build document-to-path mapping
        self._build_doc_path_mapping()

    def _build_doc_path_mapping(self):
        """Build mapping from document index to taxonomy path"""
        self.doc_to_path = {}

        def traverse(node):
            # Only assign leaf nodes to documents
            if not node.children:
                for doc_idx in node.doc_indices:
                    self.doc_to_path[doc_idx] = node

            for child in node.children:
                traverse(child)

        traverse(self.taxonomy_root)
        logger.info(f"Built path mapping for {len(self.doc_to_path)} documents")

    def get_path_text(self, doc_idx: int, format_style: str = "hierarchical") -> str:
        """
        Get formatted path text for a document

        Args:
            doc_idx: Document index
            format_style: One of ["hierarchical", "flat", "definition_only"]

        Returns:
            Formatted path text
        """
        if doc_idx not in self.doc_to_path:
            logger.warning(f"Document {doc_idx} not found in taxonomy")
            return ""

        leaf_node = self.doc_to_path[doc_idx]
        path_nodes = self.query.get_ancestors(leaf_node) + [leaf_node]
        path_nodes = path_nodes[1:]  # No Root

        if format_style == "hierarchical":
            # Format: "Root: definition > Level1: definition > ..."
            path_parts = []
            for node in path_nodes:
                parts = []
                if self.include_names:
                    parts.append(node.name)
                if self.include_definitions:
                    parts.append(node.definition)
                if parts:
                    path_parts.append(": ".join(parts))
            return " > ".join(path_parts)

        elif format_style == "flat":
            # Format: "concept1, concept2, concept3, ..."
            parts = []
            for node in path_nodes:
                if self.include_names:
                    parts.append(node.name)
                if self.include_definitions:
                    parts.append(node.definition)
            return ", ".join(parts)

        elif format_style == "definition_only":
            # Only include definitions, more concise
            definitions = [node.definition for node in path_nodes]
            return " > ".join(definitions)

        else:
            raise ValueError(f"Unknown format_style: {format_style}")

    def get_all_path_texts(self, doc_indices: List[int],
                          format_style: str = "hierarchical") -> List[str]:
        """Get path texts for multiple documents"""
        return [self.get_path_text(idx, format_style) for idx in doc_indices]

    def get_path_statistics(self) -> Dict:
        """Get statistics about paths"""
        path_lengths = []
        path_texts_lengths = []

        for doc_idx in self.doc_to_path.keys():
            leaf_node = self.doc_to_path[doc_idx]
            path_nodes = self.query.get_ancestors(leaf_node) + [leaf_node]
            path_lengths.append(len(path_nodes))

            path_text = self.get_path_text(doc_idx)
            path_texts_lengths.append(len(path_text.split()))

        return {
            'num_documents': len(self.doc_to_path),
            'avg_path_length': np.mean(path_lengths),
            'std_path_length': np.std(path_lengths),
            'min_path_length': np.min(path_lengths),
            'max_path_length': np.max(path_lengths),
            'avg_path_text_length': np.mean(path_texts_lengths),
            'std_path_text_length': np.std(path_texts_lengths),
        }


class AugmentedTextEncoder:
    """Encode text augmented with taxonomy path information"""

    def __init__(self, text_encoder: TextEncoder, path_extractor: PathTextExtractor,
                 augmentation_mode: str = "concat", pooling: str = "mean"):
        """
        Args:
            text_encoder: Pre-trained text encoder
            path_extractor: Path text extractor
            augmentation_mode: How to combine original and path text
                - "concat": Concatenate with separator
                - "separate": Encode separately and combine embeddings
            pooling: Pooling strategy for text encoder
        """
        self.text_encoder = text_encoder
        self.path_extractor = path_extractor
        self.augmentation_mode = augmentation_mode
        self.pooling = pooling

    def encode_augmented(self, doc_indices: List[int],
                        original_texts: List[str],
                        format_style: str = "hierarchical",
                        batch_size: int = 32) -> torch.Tensor:
        """
        Encode documents with augmented path information

        Args:
            doc_indices: Document indices
            original_texts: Original text for each document
            format_style: Path formatting style
            batch_size: Batch size for encoding

        Returns:
            Augmented embeddings [N, D]
        """
        if self.augmentation_mode == "concat":
            return self._encode_concatenated(
                doc_indices, original_texts, format_style, batch_size
            )
        elif self.augmentation_mode == "separate":
            return self._encode_separate(
                doc_indices, original_texts, format_style, batch_size
            )
        else:
            raise ValueError(f"Unknown augmentation_mode: {self.augmentation_mode}")

    def _encode_concatenated(self, doc_indices: List[int],
                            original_texts: List[str],
                            format_style: str,
                            batch_size: int) -> torch.Tensor:
        """Concatenate path text to original text before encoding"""
        path_texts = self.path_extractor.get_all_path_texts(doc_indices, format_style)

        # Concatenate texts
        augmented_texts = []
        for orig_text, path_text in zip(original_texts, path_texts):
            if path_text:
                orig_text = orig_text[:384]  # !!!!!
                augmented_text = f"{orig_text} [SEP] {path_text}"
            else:
                augmented_text = orig_text
            augmented_texts.append(augmented_text)

        # Encode in batches
        embeddings = []
        for i in tqdm(range(0, len(augmented_texts), batch_size),
                     desc="Encoding augmented texts"):
            batch_texts = augmented_texts[i:i + batch_size]
            with torch.no_grad():
                batch_emb = self.text_encoder(
                    input_text=batch_texts,
                    pooling=self.pooling
                )
            embeddings.append(batch_emb.cpu())

        return torch.cat(embeddings, dim=0)

    def _encode_separate(self, doc_indices: List[int],
                        original_texts: List[str],
                        format_style: str,
                        batch_size: int) -> torch.Tensor:
        """Encode original and path texts separately, then combine"""
        path_texts = self.path_extractor.get_all_path_texts(doc_indices, format_style)

        # Encode original texts
        orig_embeddings = []
        for i in tqdm(range(0, len(original_texts), batch_size),
                     desc="Encoding original texts"):
            batch_texts = original_texts[i:i + batch_size]
            with torch.no_grad():
                batch_emb = self.text_encoder(
                    input_text=batch_texts,
                    pooling=self.pooling
                )
            orig_embeddings.append(batch_emb.cpu())
        orig_embeddings = torch.cat(orig_embeddings, dim=0)

        # Encode path texts
        path_embeddings = []
        for i in tqdm(range(0, len(path_texts), batch_size),
                     desc="Encoding path texts"):
            batch_texts = path_texts[i:i + batch_size]
            # Handle empty path texts
            batch_texts = [t if t else "unknown" for t in batch_texts]
            with torch.no_grad():
                batch_emb = self.text_encoder(
                    input_text=batch_texts,
                    pooling=self.pooling
                )
            path_embeddings.append(batch_emb.cpu())
        path_embeddings = torch.cat(path_embeddings, dim=0)

        # Combine embeddings (simple concatenation)
        combined_embeddings = torch.cat([orig_embeddings, path_embeddings], dim=1)

        return combined_embeddings


class MLPClassifier(nn.Module):
    """Simple MLP classifier for frozen embeddings"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def evaluate_with_mlp(embeddings: torch.Tensor, data, device: torch.device,
                     hidden_dim: int = 256, n_layers: int = 2,
                     lr: float = 0.01, weight_decay: float = 5e-4,
                     max_epochs: int = 500, patience: int = 50,
                     n_runs: int = 5) -> Dict:
    """
    Evaluate embeddings using MLP classifier

    Args:
        embeddings: Node embeddings [N, D]
        data: Graph data with labels and masks
        device: Device for training
        ... training hyperparameters

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating with MLP classifier...")

    X = embeddings
    y = data.y.cpu()
    train_mask = data.train_mask.cpu()
    val_mask = data.val_mask.cpu()
    test_mask = data.test_mask.cpu()

    n_classes = len(torch.unique(y))
    input_dim = X.shape[1]

    all_results = {
        'val_acc': [],
        'test_acc': [],
        'test_f1_micro': [],
        'test_f1_macro': []
    }

    for run in range(n_runs):
        logger.info(f"  Run {run + 1}/{n_runs}")

        # Initialize model
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_classes,
            n_layers=n_layers,
            dropout=0.0
        ).to(device)

        X_device = X.to(device)
        y_device = y.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        # Training
        best_val_acc = 0
        best_model_state = None

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()

            logits = model(X_device)
            loss = criterion(logits[train_mask], y_device[train_mask])

            loss.backward()
            optimizer.step()

            # Validation
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_device)
                    val_pred = val_logits[val_mask].argmax(dim=1)
                    val_acc = accuracy_score(
                        y_device[val_mask].cpu().numpy(),
                        val_pred.cpu().numpy()
                    )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())

                early_stopping(1 - val_acc)  # Use negative accuracy as loss
                if early_stopping.early_stop:
                    break

        # Load best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            logits = model(X_device)

            val_pred = logits[val_mask].argmax(dim=1).cpu().numpy()
            val_true = y_device[val_mask].cpu().numpy()
            val_acc = accuracy_score(val_true, val_pred)

            test_pred = logits[test_mask].argmax(dim=1).cpu().numpy()
            test_true = y_device[test_mask].cpu().numpy()
            test_acc = accuracy_score(test_true, test_pred)
            test_f1_micro = f1_score(test_true, test_pred, average='micro')
            test_f1_macro = f1_score(test_true, test_pred, average='macro')

        all_results['val_acc'].append(val_acc)
        all_results['test_acc'].append(test_acc)
        all_results['test_f1_micro'].append(test_f1_micro)
        all_results['test_f1_macro'].append(test_f1_macro)

        logger.info(f"    Val: {val_acc:.4f}, Test: {test_acc:.4f}, "
                                      f"F1-Micro: {test_f1_micro:.4f}, F1-Macro: {test_f1_macro:.4f}")

    # Compute statistics
    results = {}
    for metric, values in all_results.items():
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)

    logger.info("\nMLP Classification Results:")
    logger.info(f"  Val Accuracy:    {results['val_acc_mean']:.4f} ± {results['val_acc_std']:.4f}")
    logger.info(f"  Test Accuracy:   {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    logger.info(f"  Test F1-Micro:   {results['test_f1_micro_mean']:.4f} ± {results['test_f1_micro_std']:.4f}")
    logger.info(f"  Test F1-Macro:   {results['test_f1_macro_mean']:.4f} ± {results['test_f1_macro_std']:.4f}")

    return results


def evaluate_with_gnn(embeddings: torch.Tensor, data, device: torch.device,
                     gnn_config: Dict, lr: float = 0.01,
                     weight_decay: float = 5e-4,
                     max_epochs: int = 500, patience: int = 50,
                     n_runs: int = 5) -> Dict:
    """
    Evaluate embeddings by training a GNN on top

    Args:
        embeddings: Initial node embeddings [N, D]
        data: Graph data with edge_index, labels, and masks
        device: Device for training
        gnn_config: GNN configuration dict
        ... training hyperparameters

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating with GNN classifier...")

    y = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    edge_index = data.edge_index

    n_classes = len(torch.unique(y))
    input_dim = embeddings.shape[1]

    all_results = {
        'val_acc': [],
        'test_acc': [],
        'test_f1_micro': [],
        'test_f1_macro': []
    }

    for run in range(n_runs):
        logger.info(f"  Run {run + 1}/{n_runs}")

        # Initialize GNN encoder + classifier
        gnn_encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=gnn_config.get('hidden_dim', 256),
            output_dim=gnn_config.get('output_dim', 128),
            n_layers=gnn_config.get('n_layers', 2),
            gnn_type=gnn_config.get('gnn_type', 'GCN'),
            dropout=gnn_config.get('dropout', 0.5),
            batch_norm=gnn_config.get('batch_norm', True),
            residual_conn=gnn_config.get('residual_conn', True),
            jump_knowledge=gnn_config.get('jump_knowledge', False)
        ).to(device)

        classifier = nn.Linear(gnn_config.get('output_dim', 128), n_classes).to(device)

        # Move embeddings to device (as input features)
        X = embeddings.to(device)

        optimizer = optim.Adam(
            list(gnn_encoder.parameters()) + list(classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        # Training
        best_val_acc = 0
        best_gnn_state = None
        best_classifier_state = None

        for epoch in range(max_epochs):
            gnn_encoder.train()
            classifier.train()
            optimizer.zero_grad()

            # Forward
            h = gnn_encoder(X, edge_index)
            logits = classifier(h)
            loss = criterion(logits[train_mask], y[train_mask])

            # Backward
            loss.backward()
            optimizer.step()

            # Validation
            if (epoch + 1) % 10 == 0:
                gnn_encoder.eval()
                classifier.eval()
                with torch.no_grad():
                    h_val = gnn_encoder(X, edge_index)
                    val_logits = classifier(h_val)
                    val_pred = val_logits[val_mask].argmax(dim=1)
                    val_acc = accuracy_score(
                        y[val_mask].cpu().numpy(),
                        val_pred.cpu().numpy()
                    )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_gnn_state = copy.deepcopy(gnn_encoder.state_dict())
                    best_classifier_state = copy.deepcopy(classifier.state_dict())

                early_stopping(1 - val_acc)
                if early_stopping.early_stop:
                    break

        # Load best model and evaluate
        if best_gnn_state is not None:
            gnn_encoder.load_state_dict(best_gnn_state)
            classifier.load_state_dict(best_classifier_state)

        gnn_encoder.eval()
        classifier.eval()
        with torch.no_grad():
            h = gnn_encoder(X, edge_index)
            logits = classifier(h)

            val_pred = logits[val_mask].argmax(dim=1).cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
            val_acc = accuracy_score(val_true, val_pred)

            test_pred = logits[test_mask].argmax(dim=1).cpu().numpy()
            test_true = y[test_mask].cpu().numpy()
            test_acc = accuracy_score(test_true, test_pred)
            test_f1_micro = f1_score(test_true, test_pred, average='micro')
            test_f1_macro = f1_score(test_true, test_pred, average='macro')

        all_results['val_acc'].append(val_acc)
        all_results['test_acc'].append(test_acc)
        all_results['test_f1_micro'].append(test_f1_micro)
        all_results['test_f1_macro'].append(test_f1_macro)

        logger.info(f"    Val: {val_acc:.4f}, Test: {test_acc:.4f}, "
                   f"F1-Micro: {test_f1_micro:.4f}, F1-Macro: {test_f1_macro:.4f}")

    # Compute statistics
    results = {}
    for metric, values in all_results.items():
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)

    logger.info("\nGNN Classification Results:")
    logger.info(f"  Val Accuracy:    {results['val_acc_mean']:.4f} ± {results['val_acc_std']:.4f}")
    logger.info(f"  Test Accuracy:   {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    logger.info(f"  Test F1-Micro:   {results['test_f1_micro_mean']:.4f} ± {results['test_f1_micro_std']:.4f}")
    logger.info(f"  Test F1-Macro:   {results['test_f1_macro_mean']:.4f} ± {results['test_f1_macro_std']:.4f}")

    return results


def compare_embeddings(baseline_emb: torch.Tensor, augmented_emb: torch.Tensor,
                      data, device: torch.device, evaluation_mode: str,
                      gnn_config: Optional[Dict] = None,
                      n_runs: int = 5) -> Dict:
    """
    Compare baseline and augmented embeddings

    Args:
        baseline_emb: Baseline embeddings (original text only)
        augmented_emb: Augmented embeddings (original + path)
        data: Graph data
        device: Device for evaluation
        evaluation_mode: "mlp" or "gnn"
        gnn_config: GNN configuration (required if mode="gnn")
        n_runs: Number of evaluation runs

    Returns:
        Comparison results
    """
    logger.info("=" * 80)
    logger.info(f"Comparing Baseline vs Augmented Embeddings ({evaluation_mode.upper()})")
    logger.info("=" * 80)

    if evaluation_mode == "mlp":
        logger.info("\n--- Baseline (Original Text Only) ---")
        set_random_seed(42)
        baseline_results = evaluate_with_mlp(
            baseline_emb, data, device, n_runs=n_runs
        )

        logger.info("\n--- Augmented (Original + Path) ---")
        set_random_seed(42)
        augmented_results = evaluate_with_mlp(
            augmented_emb, data, device, n_runs=n_runs
        )

    elif evaluation_mode == "gnn":
        if gnn_config is None:
            raise ValueError("gnn_config must be provided for GNN evaluation")

        logger.info("\n--- Baseline (Original Text Only) ---")
        set_random_seed(42)
        baseline_results = evaluate_with_gnn(
            baseline_emb, data, device, gnn_config, n_runs=n_runs
        )

        logger.info("\n--- Augmented (Original + Path) ---")
        set_random_seed(42)
        augmented_results = evaluate_with_gnn(
            augmented_emb, data, device, gnn_config, n_runs=n_runs
        )

    else:
        raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}")

    # Compute improvements
    comparison = {
        'baseline': baseline_results,
        'augmented': augmented_results,
        'improvements': {}
    }

    for metric in ['test_acc', 'test_f1_micro', 'test_f1_macro']:
        baseline_mean = baseline_results[f'{metric}_mean']
        augmented_mean = augmented_results[f'{metric}_mean']

        abs_improvement = augmented_mean - baseline_mean
        rel_improvement = (abs_improvement / baseline_mean * 100) if baseline_mean > 0 else 0

        comparison['improvements'][metric] = {
            'absolute': abs_improvement,
            'relative_pct': rel_improvement
        }

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<20} {'Baseline':<20} {'Augmented':<20} {'Improvement':<20}")
    logger.info("-" * 80)

    for metric in ['test_acc', 'test_f1_micro', 'test_f1_macro']:
        baseline_mean = baseline_results[f'{metric}_mean']
        baseline_std = baseline_results[f'{metric}_std']
        augmented_mean = augmented_results[f'{metric}_mean']
        augmented_std = augmented_results[f'{metric}_std']

        improvement = comparison['improvements'][metric]

        metric_name = metric.replace('test_', '').replace('_', '-').upper()
        baseline_str = f"{baseline_mean:.4f} ± {baseline_std:.4f}"
        augmented_str = f"{augmented_mean:.4f} ± {augmented_std:.4f}"
        improvement_str = f"+{improvement['absolute']:.4f} ({improvement['relative_pct']:+.2f}%)"

        logger.info(f"{metric_name:<20} {baseline_str:<20} {augmented_str:<20} {improvement_str:<20}")

    logger.info("=" * 80)

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate taxonomy quality via path-augmented text features'
    )

    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., cora, citeseer)')
    parser.add_argument('--data_path', type=str, default='.',
                       help='Path prefix for data')
    parser.add_argument('--taxonomy_path', type=str, required=True,
                       help='Path to taxonomy pickle file')

    # Text encoder arguments
    parser.add_argument('--text_encoder', type=str, default='roberta',
                       choices=['MiniLM', 'SentenceBert', 'e5-large', 'roberta'],
                       help='Text encoder model')
    parser.add_argument('--text_batch_size', type=int, default=32,
                       help='Batch size for text encoding')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'cls'],
                       help='Pooling strategy for text encoder')

    # Path extraction arguments
    parser.add_argument('--include_definitions', type=int, default=1,
                       help='Include concept definitions in path')
    parser.add_argument('--include_names', type=int, default=1,
                       help='Include concept names in path')
    parser.add_argument('--format_style', type=str, default='hierarchical',
                       choices=['hierarchical', 'flat', 'definition_only'],
                       help='Path text formatting style')

    # Augmentation arguments
    parser.add_argument('--augmentation_mode', type=str, default='concat',
                       choices=['concat', 'separate'],
                       help='How to combine original and path text')

    # Evaluation arguments
    parser.add_argument('--evaluation_mode', type=str, default='both',
                       choices=['mlp', 'gnn', 'both'],
                       help='Evaluation mode')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of evaluation runs')

    # GNN config (for GNN evaluation)
    parser.add_argument('--gnn_type', type=str, default='GCN',
                       help='GNN architecture')
    parser.add_argument('--gnn_hidden_dim', type=int, default=256,
                       help='GNN hidden dimension')
    parser.add_argument('--gnn_output_dim', type=int, default=128,
                       help='GNN output dimension')
    parser.add_argument('--gnn_n_layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--gnn_dropout', type=float, default=0.0,
                       help='GNN dropout rate')

    # Training arguments
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=500,
                       help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./taxonomy_eval_results/taxo_as_features',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Setup
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TAXONOMY QUALITY EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Taxonomy: {args.taxonomy_path}")
    logger.info(f"Text Encoder: {args.text_encoder}")
    logger.info(f"Format Style: {args.format_style}")
    logger.info(f"Augmentation Mode: {args.augmentation_mode}")
    logger.info(f"Evaluation Mode: {args.evaluation_mode}")
    logger.info("=" * 80)

    # Load dataset
    logger.info("\n[1/6] Loading dataset...")
    data = load_graph_dataset(
        dataset_name=args.dataset,
        device=device,
        path_prefix=args.data_path,
        re_split=0
    )
    logger.info(f"  Nodes: {data.num_nodes}")
    logger.info(f"  Edges: {data.edge_index.shape[1]}")
    logger.info(f"  Classes: {len(torch.unique(data.y))}")
    logger.info(f"  Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")

    # Load taxonomy
    logger.info("\n[2/6] Loading taxonomy...")
    taxonomy_root = TaxonomyLoader.from_pickle(args.taxonomy_path)
    query = TaxonomyQuery(taxonomy_root)
    stats = query.get_statistics()
    logger.info(f"  Total nodes: {stats['total_nodes']}")
    logger.info(f"  Leaf nodes: {stats['leaf_nodes']}")
    logger.info(f"  Max depth: {stats['max_depth']}")

    # Initialize path extractor
    logger.info("\n[3/6] Extracting taxonomy paths...")
    path_extractor = PathTextExtractor(
        taxonomy_root=taxonomy_root,
        include_definitions=bool(args.include_definitions),
        include_names=bool(args.include_names),
        separator=" [SEP] "
    )

    path_stats = path_extractor.get_path_statistics()
    logger.info(f"  Documents with paths: {path_stats['num_documents']}")
    logger.info(f"  Avg path length: {path_stats['avg_path_length']:.2f} ± {path_stats['std_path_length']:.2f}")
    logger.info(
        f"  Avg path text length: {path_stats['avg_path_text_length']:.1f} ± {path_stats['std_path_text_length']:.1f} words")

    # Example paths
    logger.info("\n  Example paths:")
    for i in range(min(3, data.num_nodes)):
        if i in path_extractor.doc_to_path:
            path_text = path_extractor.get_path_text(i, args.format_style)
            logger.info(f"    Doc {i}: {path_text[:150]}...")

    # Initialize text encoder
    logger.info("\n[4/6] Initializing text encoder...")
    encoder_type = "LM" if args.text_encoder in ["MiniLM", "SentenceBert", "e5-large", "roberta"] else "LLM"
    text_encoder = TextEncoder(
        encoder_name=args.text_encoder,
        encoder_type=encoder_type,
        device=device
    )

    # Check for cached embeddings
    emb_dir = Path(args.data_path) / "embeddings" / args.text_encoder / args.dataset
    baseline_emb_path = emb_dir / f"{args.dataset}.pt"
    taxonomy_version = Path(args.taxonomy_path).stem
    augmented_emb_path = emb_dir / f"{args.dataset}_augmented_{args.format_style}_{args.augmentation_mode}_{taxonomy_version}.pt"

    # Encode baseline embeddings (original text only)
    if baseline_emb_path.exists():
        logger.info(f"  Loading cached baseline embeddings from {baseline_emb_path}")
        baseline_embeddings = torch.load(baseline_emb_path, map_location='cpu')
    else:
        logger.info("  Encoding baseline embeddings (original text only)...")
        emb_dir.mkdir(parents=True, exist_ok=True)

        baseline_emb_list = []
        for i in tqdm(range(0, len(data.raw_texts), args.text_batch_size),
                      desc="  Encoding"):
            batch_texts = data.raw_texts[i:i + args.text_batch_size]
            with torch.no_grad():
                batch_emb = text_encoder(
                    input_text=batch_texts,
                    pooling=args.pooling
                )
            baseline_emb_list.append(batch_emb.cpu())

        baseline_embeddings = torch.cat(baseline_emb_list, dim=0)
        torch.save(baseline_embeddings, baseline_emb_path)
        logger.info(f"  Saved baseline embeddings to {baseline_emb_path}")

    logger.info(f"  Baseline embedding shape: {baseline_embeddings.shape}")

    # Encode augmented embeddings (original + path)
    if augmented_emb_path.exists():
        logger.info(f"  Loading cached augmented embeddings from {augmented_emb_path}")
        augmented_embeddings = torch.load(augmented_emb_path, map_location='cpu')
    else:
        logger.info("  Encoding augmented embeddings (original + path)...")

        augmented_text_encoder = AugmentedTextEncoder(
            text_encoder=text_encoder,
            path_extractor=path_extractor,
            augmentation_mode=args.augmentation_mode,
            pooling=args.pooling
        )

        doc_indices = list(range(data.num_nodes))
        augmented_embeddings = augmented_text_encoder.encode_augmented(
            doc_indices=doc_indices,
            original_texts=data.raw_texts,
            format_style=args.format_style,
            batch_size=args.text_batch_size
        )

        torch.save(augmented_embeddings, augmented_emb_path)
        logger.info(f"  Saved augmented embeddings to {augmented_emb_path}")

    logger.info(f"  Augmented embedding shape: {augmented_embeddings.shape}")

    # Prepare GNN config
    gnn_config = {
        'gnn_type': args.gnn_type,
        'hidden_dim': args.gnn_hidden_dim,
        'output_dim': args.gnn_output_dim,
        'n_layers': args.gnn_n_layers,
        'dropout': args.gnn_dropout,
        'batch_norm': True,
        'residual_conn': True,
        'jump_knowledge': False
    }

    # Evaluation
    all_results = {}

    if args.evaluation_mode in ['mlp', 'both']:
        logger.info("\n[5/6] Evaluating with MLP...")
        mlp_comparison = compare_embeddings(
            baseline_emb=baseline_embeddings,
            augmented_emb=augmented_embeddings,
            data=data,
            device=device,
            evaluation_mode='mlp',
            n_runs=args.n_runs
        )
        all_results['mlp'] = mlp_comparison

    if args.evaluation_mode in ['gnn', 'both']:
        logger.info("\n[5/6] Evaluating with GNN...")
        gnn_comparison = compare_embeddings(
            baseline_emb=baseline_embeddings,
            augmented_emb=augmented_embeddings,
            data=data,
            device=device,
            evaluation_mode='gnn',
            gnn_config=gnn_config,
            n_runs=args.n_runs
        )
        all_results['gnn'] = gnn_comparison

    # Save results
    logger.info("\n[6/6] Saving results...")

    # Save detailed results
    results_path = output_dir / f'taxonomy_quality_results_{args.format_style}_{args.augmentation_mode}.json'

    def convert_to_serializable(obj):
        """Convert numpy/torch types to Python native types"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(all_results)

    # Add metadata
    serializable_results['metadata'] = {
        'dataset': args.dataset,
        'taxonomy_path': args.taxonomy_path,
        'text_encoder': args.text_encoder,
        'format_style': args.format_style,
        'augmentation_mode': args.augmentation_mode,
        'evaluation_mode': args.evaluation_mode,
        'n_runs': args.n_runs,
        'path_statistics': convert_to_serializable(path_stats),
        'taxonomy_statistics': {
            'total_nodes': stats['total_nodes'],
            'leaf_nodes': stats['leaf_nodes'],
            'max_depth': stats['max_depth']
        }
    }

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"  Results saved to {results_path}")

    # Generate summary report
    summary_path = output_dir / f'taxonomy_quality_summary_{args.format_style}_{args.augmentation_mode}.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TAXONOMY QUALITY EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Taxonomy: {args.taxonomy_path}\n")
        f.write(f"Text Encoder: {args.text_encoder}\n")
        f.write(f"Format Style: {args.format_style}\n")
        f.write(f"Augmentation Mode: {args.augmentation_mode}\n\n")

        f.write("Taxonomy Statistics:\n")
        f.write(f"  Total nodes: {stats['total_nodes']}\n")
        f.write(f"  Leaf nodes: {stats['leaf_nodes']}\n")
        f.write(f"  Max depth: {stats['max_depth']}\n")
        f.write(f"  Documents with paths: {path_stats['num_documents']}\n")
        f.write(f"  Avg path length: {path_stats['avg_path_length']:.2f} ± {path_stats['std_path_length']:.2f}\n\n")

        for eval_mode in ['mlp', 'gnn']:
            if eval_mode in all_results:
                comparison = all_results[eval_mode]

                f.write("=" * 80 + "\n")
                f.write(f"{eval_mode.upper()} EVALUATION RESULTS\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"{'Metric':<20} {'Baseline':<25} {'Augmented':<25} {'Improvement':<20}\n")
                f.write("-" * 90 + "\n")

                for metric in ['test_acc', 'test_f1_micro', 'test_f1_macro']:
                    baseline_mean = comparison['baseline'][f'{metric}_mean']
                    baseline_std = comparison['baseline'][f'{metric}_std']
                    augmented_mean = comparison['augmented'][f'{metric}_mean']
                    augmented_std = comparison['augmented'][f'{metric}_std']
                    improvement = comparison['improvements'][metric]

                    metric_name = metric.replace('test_', '').replace('_', '-').upper()
                    baseline_str = f"{baseline_mean:.4f} ± {baseline_std:.4f}"
                    augmented_str = f"{augmented_mean:.4f} ± {augmented_std:.4f}"
                    improvement_str = f"+{improvement['absolute']:.4f} ({improvement['relative_pct']:+.2f}%)"

                    f.write(f"{metric_name:<20} {baseline_str:<25} {augmented_str:<25} {improvement_str:<20}\n")

                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        # Analyze improvements
        for eval_mode in ['mlp', 'gnn']:
            if eval_mode in all_results:
                comparison = all_results[eval_mode]
                f.write(f"{eval_mode.upper()} Mode:\n")

                for metric in ['test_acc', 'test_f1_micro', 'test_f1_macro']:
                    improvement = comparison['improvements'][metric]
                    metric_name = metric.replace('test_', '').replace('_', ' ').title()

                    if improvement['relative_pct'] > 0:
                        f.write(f"  ✓ {metric_name}: +{improvement['relative_pct']:.2f}% improvement\n")
                    else:
                        f.write(f"  ✗ {metric_name}: {improvement['relative_pct']:.2f}% (no improvement)\n")

                f.write("\n")

        f.write("Interpretation:\n")
        f.write("  - Positive improvements indicate that taxonomy paths provide useful semantic information\n")
        f.write("  - Larger improvements suggest the taxonomy captures high-level concepts not easily\n")
        f.write("    extractible from raw text alone\n")
        f.write("  - GNN improvements indicate taxonomy helps with structural learning\n")

    logger.info(f"  Summary saved to {summary_path}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 80)

    if 'mlp' in all_results:
        mlp_acc_improvement = all_results['mlp']['improvements']['test_acc']['relative_pct']
        logger.info(f"MLP Accuracy Improvement: {mlp_acc_improvement:+.2f}%")

    if 'gnn' in all_results:
        gnn_acc_improvement = all_results['gnn']['improvements']['test_acc']['relative_pct']
        logger.info(f"GNN Accuracy Improvement: {gnn_acc_improvement:+.2f}%")
    logger.info("=" * 80)

    # Generate visualization comparing improvements
    logger.info("\nGenerating improvement visualization...")
    plot_path = output_dir / f'taxonomy_quality_comparison_{args.format_style}_{args.augmentation_mode}.png'

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        metrics_display = {
            'test_acc': 'Accuracy',
            'test_f1_micro': 'F1-Micro',
            'test_f1_macro': 'F1-Macro'
        }

        for idx, eval_mode in enumerate(['mlp', 'gnn']):
            if eval_mode in all_results:
                ax = axes[idx]
                comparison = all_results[eval_mode]

                metrics = list(metrics_display.keys())
                x_pos = np.arange(len(metrics))

                baseline_means = [comparison['baseline'][f'{m}_mean'] for m in metrics]
                baseline_stds = [comparison['baseline'][f'{m}_std'] for m in metrics]
                augmented_means = [comparison['augmented'][f'{m}_mean'] for m in metrics]
                augmented_stds = [comparison['augmented'][f'{m}_std'] for m in metrics]

                width = 0.35

                bars1 = ax.bar(x_pos - width / 2, baseline_means, width,
                               yerr=baseline_stds, label='Baseline (Original Text)',
                               capsize=5, alpha=0.8, color='skyblue')
                bars2 = ax.bar(x_pos + width / 2, augmented_means, width,
                               yerr=augmented_stds, label='Augmented (Original + Path)',
                               capsize=5, alpha=0.8, color='lightcoral')

                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

                # Add improvement percentages above
                for i, metric in enumerate(metrics):
                    improvement_pct = comparison['improvements'][metric]['relative_pct']
                    y_pos = max(augmented_means[i], baseline_means[i]) + max(augmented_stds[i], baseline_stds[i]) + 0.02
                    color = 'green' if improvement_pct > 0 else 'red'
                    ax.text(i, y_pos, f'{improvement_pct:+.2f}%',
                            ha='center', va='bottom', fontsize=10,
                            fontweight='bold', color=color)

                ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
                ax.set_ylabel('Score', fontsize=12, fontweight='bold')
                ax.set_title(f'{eval_mode.upper()} Evaluation', fontsize=14, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([metrics_display[m] for m in metrics])
                ax.legend(loc='lower right', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(0, 1.0)

        plt.suptitle(f'Taxonomy Quality Evaluation: {args.dataset.upper()}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Visualization saved to {plot_path}")

    except ImportError:
        logger.warning("  Matplotlib not available, skipping visualization")
    except Exception as e:
        logger.warning(f"  Error generating visualization: {e}")

    logger.info("\nAll results saved to: " + str(output_dir))
    logger.info("✓ Evaluation complete!")


if __name__ == '__main__':
    main()
