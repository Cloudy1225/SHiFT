"""
model.py

Hierarchical TAG learning with LLM-guided taxonomy
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch_geometric.utils import negative_sampling, to_undirected, add_remaining_self_loops

from dataloader import load_graph_dataset
from gnn import GNNEncoder
from lm import TextEncoder
from taxonomy import TaxonomySaver
from taxonomy_initialize import TaxonomyBuilder, LLMClient, CoreSetSampler
from taxonomy_update import TaxonomyUpdater
from utils import EarlyStopping

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HierarchicalTAGLearner:
    """Main class for hierarchical TAG learning"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # Load dataset
        logger.info(f"Loading dataset: {args.dataset}")
        self.data = load_graph_dataset(
            dataset_name=args.dataset,
            device=self.device,
            path_prefix=args.data_path,
            re_split=args.re_split
        )

        # Ensure undirected graph
        self.data.edge_index = to_undirected(self.data.edge_index)

        # Initialize text encoder for getting initial embeddings
        self._initialize_text_embeddings()

        # Initialize GNN encoder
        self.gnn_encoder = GNNEncoder(
            input_dim=self.data.x.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            n_layers=args.n_layers,
            gnn_type=args.gnn_type,
            dropout=args.dropout,
            batch_norm=args.batch_norm,
            residual_conn=args.residual_conn,
            jump_knowledge=args.jump_knowledge
        ).to(self.device)

        # Initialize projection head for TCA
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(args.output_dim, args.projection_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.projection_dim, args.text_emb_dim)
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.projection_head.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Taxonomy components (initialized later)
        self.taxonomy_root = None
        self.llm_client = None
        self.sampler = None
        self.taxonomy_builder = None
        self.taxonomy_updater = None
        self.previous_embeddings = None

        # Statistics
        self.current_epoch = 0
        self.best_loss = float('inf')

        # Cache management
        self._leaf_assignments_cache = None
        self._doc_paths_cache = None
        self._cophenetic_distance_cache = None
        self._taxonomy_version = None

    def _initialize_text_embeddings(self):
        """Initialize or load pre-computed text embeddings"""
        emb_dir = Path(self.args.data_path) / "datasets" / self.args.text_encoder
        emb_path = emb_dir / f"{self.args.dataset}.pt"

        if emb_path.exists():
            logger.info(f"Loading pre-computed text embeddings from {emb_path}")
            text_emb = torch.load(emb_path, map_location='cpu')
        else:
            logger.info(f"Computing text embeddings with {self.args.text_encoder}")
            emb_dir.mkdir(parents=True, exist_ok=True)

            # Initialize text encoder
            encoder_type = "LM" if self.args.text_encoder in ["MiniLM", "SentenceBert", "e5-large", "roberta"] else "LLM"
            text_encoder = TextEncoder(
                encoder_name=self.args.text_encoder,
                encoder_type=encoder_type,
                device=self.device
            )

            # Compute embeddings in batches
            text_emb_list = []
            raw_texts = self.data.raw_texts
            batch_size = self.args.text_batch_size

            with torch.no_grad():
                for i in range(0, len(raw_texts), batch_size):
                    batch_texts = raw_texts[i:i + batch_size]
                    batch_emb = text_encoder(
                        input_text=batch_texts,
                        pooling="cls" if self.args.use_cls else "mean"
                    )
                    text_emb_list.append(batch_emb.cpu())

            text_emb = torch.cat(text_emb_list, dim=0)

            # Save embeddings
            torch.save(text_emb, emb_path)
            logger.info(f"Saved text embeddings to {emb_path}")

        # Set as node features
        self.data.x = text_emb.to(self.device)
        self.args.text_emb_dim = text_emb.shape[1]
        logger.info(f"Text embedding shape: {text_emb.shape}")

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

    def warmup_pretrain(self):
        """Graph contrastive pretraining (warmup phase)"""
        logger.info("=" * 60)
        logger.info("Starting Warmup Pretraining")
        logger.info("=" * 60)

        early_stopping = EarlyStopping(patience=self.args.warmup_patience, verbose=True)

        for epoch in range(self.args.warmup_epochs):
        ### for epoch in range(0):
            self.gnn_encoder.train()
            self.optimizer.zero_grad()

            # Generate two augmented views
            edge_index_1 = self._augment_graph(self.data.edge_index, drop_edge_p=self.args.drop_edge_p)
            edge_index_2 = self._augment_graph(self.data.edge_index, drop_edge_p=self.args.drop_edge_p)

            x_1 = self._augment_features(self.data.x, drop_feat_p=self.args.drop_feat_p)
            x_2 = self._augment_features(self.data.x, drop_feat_p=self.args.drop_feat_p)

            # Forward pass
            z_1 = self.gnn_encoder(x_1, edge_index_1)
            z_2 = self.gnn_encoder(x_2, edge_index_2)

            # Normalize embeddings
            z_1 = F.normalize(z_1, p=2, dim=1)
            z_2 = F.normalize(z_2, p=2, dim=1)

            # Compute structure-aware contrastive loss
            loss = self._compute_scl_loss(z_1, z_2, self.data.edge_index)

            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % self.args.log_interval == 0:
                logger.info(f"Warmup Epoch {epoch+1}/{self.args.warmup_epochs}, Loss: {loss.item():.4f}")

            # Early stopping
            early_stopping(loss.item())
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info("Warmup pretraining completed!")

    def _augment_graph(self, edge_index, drop_edge_p=0.1):
        """Augment graph by randomly dropping edges"""
        num_edges = edge_index.shape[1]
        mask = torch.rand(num_edges, device=self.device) > drop_edge_p
        return edge_index[:, mask]

    def _augment_features(self, x, drop_feat_p=0.1):
        """Augment features by randomly masking"""
        mask = torch.rand_like(x) > drop_feat_p
        return x * mask.float()

    def _compute_scl_loss(self, z_1, z_2, edge_index):
        """
        Compute structure-aware contrastive loss (SCL)
        Uses efficient positive/negative sampling to avoid dense matrix
        """
        # Positive pairs: connected nodes (including self-loops)
        # Add self-loops
        num_nodes = z_1.shape[0]
        pos_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)

        # Sample subset of positive pairs to save memory
        num_pos = pos_edge_index.shape[1]
        if num_pos > self.args.max_pos_samples:
            perm = torch.randperm(num_pos, device=self.device)[:self.args.max_pos_samples]
            pos_edge_index = pos_edge_index[:, perm]

        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=min(self.args.max_neg_samples, num_pos)
        )

        # Compute positive loss
        pos_i, pos_j = pos_edge_index[0], pos_edge_index[1]
        pos_sim = (z_1[pos_i] * z_2[pos_j]).sum(dim=1)
        pos_loss = -pos_sim.mean()

        # Compute negative loss
        neg_i, neg_j = neg_edge_index[0], neg_edge_index[1]
        neg_sim = (z_1[neg_i] * z_2[neg_j]).sum(dim=1)
        neg_loss = self.args.gamma * (neg_sim ** 2).mean()

        return pos_loss + neg_loss

    def build_initial_taxonomy(self):
        """Build initial taxonomy tree using LLM"""
        logger.info("=" * 60)
        logger.info("Building Initial Taxonomy")
        logger.info("=" * 60)

        # Initialize components if not done
        if self.llm_client is None:
            self._initialize_taxonomy_components()

        # Get current embeddings
        self.gnn_encoder.eval()
        with torch.no_grad():
            embeddings = self.gnn_encoder(self.data.x, self.data.edge_index)
            embeddings = embeddings.cpu().numpy()

        # Normalize embeddings
        embeddings = normalize(embeddings, norm='l2')

        # Build taxonomy
        self.taxonomy_root = self.taxonomy_builder.build_taxonomy(
            embeddings=embeddings,
            raw_texts=self.data.raw_texts,
            edge_index=self.data.edge_index.cpu().numpy()
        )

        # Save taxonomy
        self._save_taxonomy(suffix="initial")

        # Store embeddings for future updates
        self.previous_embeddings = embeddings

        logger.info("Initial taxonomy built successfully!")

    def _save_taxonomy(self, suffix=""):
        """Save taxonomy to disk"""
        output_dir = Path(self.args.output_dir) / self.args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_suffix = f"_{suffix}" if suffix else ""

        TaxonomySaver.to_pickle(
            self.taxonomy_root,
            output_path=str(output_dir / f"taxonomy{filename_suffix}.pkl")
        )
        TaxonomySaver.to_json(
            self.taxonomy_root,
            output_path=str(output_dir / f"taxonomy{filename_suffix}.json")
        )
        TaxonomySaver.to_assignments(
            self.taxonomy_root,
            output_path=str(output_dir / f"assignments{filename_suffix}.tsv")
        )

        logger.info(f"Taxonomy saved to {output_dir}")

    def hierarchy_aware_training(self):
        """Main training loop with hierarchy-aware losses"""
        logger.info("=" * 60)
        logger.info("Starting Hierarchy-Aware Training")
        logger.info("=" * 60)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Precompute topic embeddings for TCA
        self._precompute_topic_embeddings()

        total_epochs = self.args.total_epochs - self.args.warmup_epochs

        for epoch in range(total_epochs):
            self.current_epoch = self.args.warmup_epochs + epoch

            # Training step
            self.gnn_encoder.train()
            self.projection_head.train()
            self.optimizer.zero_grad()

            # Generate two augmented views
            edge_index_1 = self._augment_graph(self.data.edge_index, drop_edge_p=self.args.drop_edge_p)
            edge_index_2 = self._augment_graph(self.data.edge_index, drop_edge_p=self.args.drop_edge_p)

            x_1 = self._augment_features(self.data.x, drop_feat_p=self.args.drop_feat_p)
            x_2 = self._augment_features(self.data.x, drop_feat_p=self.args.drop_feat_p)

            # Forward pass
            z_1 = self.gnn_encoder(x_1, edge_index_1)
            z_2 = self.gnn_encoder(x_2, edge_index_2)

            # Normalize embeddings
            z_1 = F.normalize(z_1, p=2, dim=1)
            z_2 = F.normalize(z_2, p=2, dim=1)

            # Compute losses
            loss_tpa = self._compute_tpa_loss(z_1, z_2)
            loss_tsa = self._compute_tsa_loss(z_1, z_2)
            loss_tca = self._compute_tca_loss(z_1, z_2)

            # Combined loss
            total_loss = loss_tpa + self.args.alpha * loss_tsa + self.args.beta * loss_tca

            total_loss.backward()
            self.optimizer.step()

            # Logging
            if (epoch + 1) % self.args.log_interval == 0:
                logger.info(
                    f"Epoch {self.current_epoch+1}/{self.args.total_epochs}, "
                    f"Total Loss: {total_loss.item():.4f}, "
                    f"tpa: {loss_tpa.item():.4f}, "
                    f"TSA: {loss_tsa.item():.4f}, "
                    f"TCA: {loss_tca.item():.4f}"
                )

            # Taxonomy update
            if (epoch + 1) % self.args.taxonomy_update_interval == 0:
                self.evaluate(finetune=self.args.finetune)
                self._update_taxonomy()

            # Early stopping
            early_stopping(total_loss.item())
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {self.current_epoch+1}")
                break

        logger.info("Hierarchy-aware training completed!")

        # Save final model and taxonomy
        self._save_model()
        self._save_taxonomy(suffix="final")

    def _precompute_topic_embeddings(self):
        """Precompute embeddings for all topic descriptions in taxonomy"""
        logger.info("Precomputing topic embeddings for TCA...")

        encoder_type = "LM" if self.args.text_encoder in ["MiniLM", "SentenceBert", "e5-large", "roberta"] else "LLM"
        text_encoder = TextEncoder(
            encoder_name=self.args.text_encoder,
            encoder_type=encoder_type,
            device=self.device
        )

        # Collect all topic descriptions
        self.topic_embeddings = {}

        def collect_topics(node):
            topic_text = f"{node.name}: {node.definition}"
            with torch.no_grad():
                topic_emb = text_encoder(
                    input_text=topic_text,
                    pooling="cls" if self.args.use_cls else "mean"
                )
            self.topic_embeddings[id(node)] = topic_emb.squeeze(0)

            for child in node.children:
                collect_topics(child)

        collect_topics(self.taxonomy_root)
        logger.info(f"Computed {len(self.topic_embeddings)} topic embeddings")

    def _compute_tpa_loss(self, z_1, z_2):
        """
        Compute Taxonomy Partition Alignment loss
        Treats nodes in same leaf cluster as positive pairs

        Improvements:
        1. Include original graph edges + self-loops as base positive pairs
        2. Vectorized generation of leaf cluster pairs (faster than nested loops)
        3. Cache leaf assignments to avoid recomputation
        """
        num_nodes = z_1.shape[0]

        # Base positive pairs: graph edges + self-loops
        base_pos_edge_index, _ = add_remaining_self_loops(self.data.edge_index, num_nodes=num_nodes)

        # Get leaf assignments (with caching)
        leaf_assignments = self._get_leaf_assignments_cached()

        # Generate positive pairs from leaf clusters (vectorized)
        leaf_pos_pairs = self._generate_leaf_pairs_vectorized(leaf_assignments)

        if len(leaf_pos_pairs) > 0:
            # Combine base pairs with leaf cluster pairs
            leaf_pos_edge_index = torch.tensor(leaf_pos_pairs, device=self.device).t()
            pos_edge_index = torch.cat([base_pos_edge_index, leaf_pos_edge_index], dim=1)
        else:
            pos_edge_index = base_pos_edge_index

        # Remove duplicate edges
        pos_edge_index = torch.unique(pos_edge_index, dim=1)

        # Sample subset to save memory
        if pos_edge_index.shape[1] > self.args.max_pos_samples:
            perm = torch.randperm(pos_edge_index.shape[1], device=self.device)[:self.args.max_pos_samples]
            pos_edge_index = pos_edge_index[:, perm]

        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=min(self.args.max_neg_samples, pos_edge_index.shape[1])
        )

        # Compute loss
        pos_i, pos_j = pos_edge_index[0], pos_edge_index[1]
        pos_sim = (z_1[pos_i] * z_2[pos_j]).sum(dim=1)
        pos_loss = -pos_sim.mean()

        neg_i, neg_j = neg_edge_index[0], neg_edge_index[1]
        neg_sim = (z_1[neg_i] * z_2[neg_j]).sum(dim=1)
        neg_loss = self.args.gamma * (neg_sim ** 2).mean()

        return pos_loss + neg_loss

    def _get_leaf_assignments_cached(self):
        """
        Get document assignments to leaf clusters with caching
        Cache is invalidated when taxonomy is updated
        """
        # Check if cache exists and is valid
        if (hasattr(self, '_leaf_assignments_cache') and
                hasattr(self, '_taxonomy_version') and
                self._taxonomy_version == self._get_taxonomy_version()):
            if self._leaf_assignments_cache is not None:
                return self._leaf_assignments_cache

        # Recompute leaf assignments
        leaf_assignments = {}

        def collect_leaves(node):
            if not node.children:  # Leaf node
                leaf_assignments[id(node)] = set(node.doc_indices)
            else:
                for child in node.children:
                    collect_leaves(child)

        collect_leaves(self.taxonomy_root)

        # Update cache
        self._leaf_assignments_cache = leaf_assignments
        self._taxonomy_version = self._get_taxonomy_version()

        return leaf_assignments

    def _get_taxonomy_version(self):
        """
        Generate a version identifier for the current taxonomy
        Used to invalidate caches when taxonomy changes
        """
        # Use tuple of (node_id, num_children) for all nodes as version
        version_info = []

        def collect_version(node):
            version_info.append((id(node), len(node.children)))
            for child in node.children:
                collect_version(child)

        if self.taxonomy_root:
            collect_version(self.taxonomy_root)

        return tuple(version_info)

    def _generate_leaf_pairs_vectorized(self, leaf_assignments):
        """
        Generate positive pairs from leaf clusters using vectorized operations
        Much faster than nested loops for large clusters

        Args:
            leaf_assignments: Dict mapping leaf_id -> set of doc_indices

        Returns:
            List of (doc_i, doc_j) pairs
        """
        all_pairs = []
        max_pairs_per_cluster = self.args.max_pos_samples // max(len(leaf_assignments), 1)

        for leaf_docs in leaf_assignments.values():
            if len(leaf_docs) < 2:
                continue

            docs = np.array(list(leaf_docs))
            n = len(docs)

            # Method 1: For small clusters, generate all pairs
            if n <= 50:
                # Use numpy meshgrid for vectorized pair generation
                i_indices, j_indices = np.triu_indices(n, k=1)
                pairs = np.stack([docs[i_indices], docs[j_indices]], axis=1)

            # Method 2: For large clusters, random sampling
            else:
                # Randomly sample pairs
                num_pairs = min(n * (n - 1) // 2, max_pairs_per_cluster)

                # Generate random pairs efficiently
                pairs = []
                attempts = 0
                max_attempts = num_pairs * 3  # Avoid infinite loop

                while len(pairs) < num_pairs and attempts < max_attempts:
                    # Randomly sample indices
                    batch_size = min(num_pairs - len(pairs), 1000)
                    i = np.random.randint(0, n, size=batch_size)
                    j = np.random.randint(0, n, size=batch_size)

                    # Keep only i < j (avoid duplicates and self-pairs)
                    valid = i < j
                    if valid.sum() > 0:
                        batch_pairs = np.stack([docs[i[valid]], docs[j[valid]]], axis=1)
                        pairs.append(batch_pairs)

                    attempts += batch_size

                if pairs:
                    pairs = np.concatenate(pairs, axis=0)
                else:
                    pairs = np.array([]).reshape(0, 2)

            # Limit pairs per cluster
            if len(pairs) > max_pairs_per_cluster:
                indices = np.random.choice(len(pairs), max_pairs_per_cluster, replace=False)
                pairs = pairs[indices]

            all_pairs.append(pairs)

        # Concatenate all pairs
        if all_pairs:
            all_pairs = np.concatenate(all_pairs, axis=0)
            return all_pairs.tolist()
        else:
            return []

    def _invalidate_taxonomy_cache(self):
        """
        Invalidate all taxonomy-related caches
        Call this after taxonomy update
        """
        if hasattr(self, '_leaf_assignments_cache'):
            delattr(self, '_leaf_assignments_cache')
        if hasattr(self, '_taxonomy_version'):
            delattr(self, '_taxonomy_version')
        if hasattr(self, '_doc_paths_cache'):
            delattr(self, '_doc_paths_cache')
        if hasattr(self, '_cophenetic_distance_cache'):
            delattr(self, '_cophenetic_distance_cache')

    def _update_taxonomy(self):
        """Incrementally update taxonomy tree"""
        logger.info(f"Updating taxonomy at epoch {self.current_epoch + 1}")

        # Get current embeddings
        self.gnn_encoder.eval()
        with torch.no_grad():
            new_embeddings = self.gnn_encoder(self.data.x, self.data.edge_index)
            new_embeddings = new_embeddings.cpu().numpy()

        # Normalize
        new_embeddings = normalize(new_embeddings, norm='l2')

        # Update taxonomy
        self.taxonomy_root = self.taxonomy_updater.update_taxonomy(
            root=self.taxonomy_root,
            old_embeddings=self.previous_embeddings,
            new_embeddings=new_embeddings,
            raw_texts=self.data.raw_texts,
            edge_index=self.data.edge_index.cpu().numpy()
        )

        # Update previous embeddings
        self.previous_embeddings = new_embeddings

        # Invalidate caches
        self._invalidate_taxonomy_cache()

        # Recompute topic embeddings
        self._precompute_topic_embeddings()

        # Save updated taxonomy
        self._save_taxonomy(suffix=f"epoch_{self.current_epoch + 1}")

        # Switch back to training mode
        self.gnn_encoder.train()

    def _compute_tsa_loss(self, z_1, z_2):
        """
        Compute Taxonomy Skeleton Alignment loss
        Uses Cophenetic Correlation Coefficient (CCC)
        """
        # Get leaf prototypes
        leaf_prototypes_1, leaf_prototypes_2 = self._compute_leaf_prototypes(z_1, z_2)

        if len(leaf_prototypes_1) < 2:
            return torch.tensor(0.0, device=self.device)

        # Stack prototypes
        P_1 = torch.stack(leaf_prototypes_1)
        P_2 = torch.stack(leaf_prototypes_2)

        # Compute pairwise distances (only upper triangular)
        # pdist returns condensed distance vector (upper triangular without diagonal)
        d_1 = torch.pdist(P_1, p=2)  # Shape: (k*(k-1)/2,)
        d_2 = torch.pdist(P_2, p=2)  # Shape: (k*(k-1)/2,)

        # Get cophenetic distances from taxonomy (also in condensed form)
        d_coph = self._compute_cophenetic_distances()

        # Compute CCC for both views
        ccc_1 = self._compute_ccc(d_1, d_coph)
        ccc_2 = self._compute_ccc(d_2, d_coph)

        # Loss: maximize CCC (minimize negative CCC)
        loss = -(ccc_1 + ccc_2) / 2.0

        return loss

    def _compute_leaf_prototypes(self, z_1, z_2):
        """Compute prototype vectors for each leaf cluster"""
        leaf_assignments = self._get_leaf_assignments_cached()

        prototypes_1 = []
        prototypes_2 = []

        for leaf_docs in leaf_assignments.values():
            if len(leaf_docs) == 0:
                continue
            docs = list(leaf_docs)
            proto_1 = z_1[docs].mean(dim=0)
            proto_2 = z_2[docs].mean(dim=0)
            prototypes_1.append(proto_1)
            prototypes_2.append(proto_2)

        return prototypes_1, prototypes_2

    def _tree_distance(self, node_id_1, node_id_2):
        """Compute tree distance (path length) between two nodes"""
        # Find nodes by id
        node_1 = self._find_node_by_id(self.taxonomy_root, node_id_1)
        node_2 = self._find_node_by_id(self.taxonomy_root, node_id_2)

        if node_1 is None or node_2 is None:
            return 0

        # Get paths to root
        path_1 = self._get_path_to_root(node_1)
        path_2 = self._get_path_to_root(node_2)

        # Find lowest common ancestor (LCA)
        lca_depth = 0
        for i in range(min(len(path_1), len(path_2))):
            if path_1[i] == path_2[i]:
                lca_depth = i + 1
            else:
                break

        # Distance = depth(node1) + depth(node2) - 2 * depth(LCA)
        distance = (len(path_1) - lca_depth) + (len(path_2) - lca_depth)

        return distance

    def _find_node_by_id(self, node, target_id):
        """Find node by its id in tree"""
        if id(node) == target_id:
            return node
        for child in node.children:
            result = self._find_node_by_id(child, target_id)
            if result:
                return result
        return None

    def _get_path_to_root(self, node):
        """Get path from node to root"""
        path = []
        current = node
        while current:
            path.append(id(current))
            current = current.parent
        return list(reversed(path))

    def _compute_ccc(self, d, d_coph):
        """
        Compute Cophenetic Correlation Coefficient from condensed distance vectors

        Args:
            d: Condensed distance vector from pdist (upper triangular)
            d_coph: Condensed cophenetic distance vector

        Returns:
            Pearson correlation coefficient
        """
        # Both vectors are already flattened upper triangular entries
        # No need for masking

        # Compute Pearson correlation
        d_mean = d.mean()
        d_coph_mean = d_coph.mean()

        d_centered = d - d_mean
        d_coph_centered = d_coph - d_coph_mean

        numerator = (d_centered * d_coph_centered).sum()
        denominator = torch.norm(d_centered, p=2) * torch.norm(d_coph_centered, p=2)

        if denominator < 1e-8:
            return torch.tensor(0.0, device=self.device)

        ccc = numerator / (denominator + 1e-8)

        return ccc

    def _compute_tca_loss(self, z_1, z_2):
        """
        Compute Taxonomy Concept Alignment loss
        Aligns node embeddings with topic description embeddings along hierarchical paths
        """
        # Project embeddings to alignment space
        proj_1 = self.projection_head(z_1)
        proj_2 = self.projection_head(z_2)

        # Normalize
        proj_1 = F.normalize(proj_1, p=2, dim=1)
        proj_2 = F.normalize(proj_2, p=2, dim=1)

        # Get document to path mappings
        doc_paths = self._get_document_paths()

        total_loss = 0.0
        count = 0

        # Sample subset of documents to save computation
        doc_indices = list(doc_paths.keys())
        if len(doc_indices) > self.args.max_tca_samples:
            doc_indices = np.random.choice(doc_indices, self.args.max_tca_samples, replace=False)

        for doc_idx in doc_indices:
            path = doc_paths[doc_idx]
            if not path:
                continue

            # Compute multi-scale alignment loss
            for level, node_id in enumerate(path):
                if node_id not in self.topic_embeddings:
                    continue

                topic_emb = self.topic_embeddings[node_id].to(self.device)
                topic_emb = F.normalize(topic_emb, p=2, dim=0)

                # Layer-dependent weight (higher weight for deeper levels)
                depth = len(path)
                weight = self.args.lambda_decay ** (depth - level - 1)

                # L2 distance loss
                loss_1 = weight * torch.norm(proj_1[doc_idx] - topic_emb, p=2)
                loss_2 = weight * torch.norm(proj_2[doc_idx] - topic_emb, p=2)

                total_loss += (loss_1 + loss_2)
                count += 2

        if count == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / count

    def _get_document_paths(self):
        """
        Get hierarchical path for each document (with caching)
        Cache is invalidated when taxonomy changes
        """
        # Check cache
        if (hasattr(self, '_doc_paths_cache') and
                hasattr(self, '_taxonomy_version') and
                self._taxonomy_version == self._get_taxonomy_version()):
            if self._doc_paths_cache is not None:
                return self._doc_paths_cache

        # Recompute document paths
        doc_paths = {}

        def traverse(node, path):
            current_path = path + [id(node)]

            if not node.children:  # Leaf node
                for doc_idx in node.doc_indices:
                    doc_paths[doc_idx] = current_path
            else:
                for child in node.children:
                    traverse(child, current_path)

        traverse(self.taxonomy_root, [])

        # Update cache
        self._doc_paths_cache = doc_paths

        return doc_paths

    def _compute_cophenetic_distances(self):
        """
        Compute cophenetic distances in condensed form (matching pdist output)
        Cache is invalidated when taxonomy changes

        Returns:
            Condensed distance vector (upper triangular without diagonal)
        """
        # Check cache
        if (hasattr(self, '_cophenetic_distance_cache') and
                hasattr(self, '_taxonomy_version') and
                self._taxonomy_version == self._get_taxonomy_version()):
            if self._cophenetic_distance_cache is not None:
                return self._cophenetic_distance_cache

        # Recompute cophenetic distances
        leaf_assignments = self._get_leaf_assignments_cached()
        leaf_ids = list(leaf_assignments.keys())
        k = len(leaf_ids)

        if k < 2:
            return torch.tensor([], device=self.device)

        # Precompute all paths to root for efficiency
        leaf_paths = {}
        for leaf_id in leaf_ids:
            node = self._find_node_by_id(self.taxonomy_root, leaf_id)
            if node:
                leaf_paths[leaf_id] = self._get_path_to_root(node)

        # Compute condensed distance vector (upper triangular)
        # Size: k * (k-1) / 2
        distances = []

        for i in range(k):
            for j in range(i + 1, k):
                leaf_i = leaf_ids[i]
                leaf_j = leaf_ids[j]

                path_i = leaf_paths.get(leaf_i, [])
                path_j = leaf_paths.get(leaf_j, [])

                # Find LCA depth
                lca_depth = 0
                for idx in range(min(len(path_i), len(path_j))):
                    if path_i[idx] == path_j[idx]:
                        lca_depth = idx + 1
                    else:
                        break

                # Distance = depth(i) + depth(j) - 2 * depth(LCA)
                distance = (len(path_i) - lca_depth) + (len(path_j) - lca_depth)
                distances.append(distance)

        # Convert to tensor
        d_coph = torch.tensor(distances, dtype=torch.float32, device=self.device)

        # Cache result
        self._cophenetic_distance_cache = d_coph

        return d_coph

    def _save_model(self):
        """Save trained model"""
        output_dir = Path(self.args.output_dir) / self.args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'gnn_encoder': self.gnn_encoder.state_dict(),
            'projection_head': self.projection_head.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'args': vars(self.args)
        }

        torch.save(checkpoint, output_dir / 'model_final.pt')
        logger.info(f"Model saved to {output_dir / 'model_final.pt'}")

    def load_model(self, checkpoint_path):
        """Load trained model"""
        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
        self.projection_head.load_state_dict(checkpoint['projection_head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_epoch = checkpoint['epoch']

        logger.info(f"Model loaded from epoch {self.current_epoch}")

    def evaluate(self, finetune=True):
        """
        Evaluate learned representations on downstream tasks

        Args:
            finetune: If True, fine-tune the GNN backbone on downstream task
        """
        logger.info("=" * 60)
        logger.info("Evaluating Learned Representations")
        if finetune:
            logger.info("Mode: Fine-tuning GNN Backbone")
        else:
            logger.info("Mode: Frozen Features + MLP Classifier")
        logger.info("=" * 60)

        results = {}

        # Node classification
        if hasattr(self.data, 'y'):
            logger.info("\n" + "-" * 60)
            logger.info("Node Classification Evaluation (5 runs)")
            logger.info("-" * 60)

            if finetune:
                classification_results = self._evaluate_classification_finetune(n_runs=5)
            else:
                # Get frozen embeddings
                self.gnn_encoder.eval()
                with torch.no_grad():
                    embeddings = self.gnn_encoder(self.data.x, self.data.edge_index)
                classification_results = self._evaluate_classification_frozen(embeddings, n_runs=5)

            results.update(classification_results)

        # Clustering evaluation (always use frozen embeddings)
        if hasattr(self.data, 'y'):
            logger.info("\n" + "-" * 60)
            logger.info("Clustering Evaluation (1 runs)")
            logger.info("-" * 60)

            self.gnn_encoder.eval()
            with torch.no_grad():
                embeddings = self.gnn_encoder(self.data.x, self.data.edge_index)

            clustering_results = self._evaluate_clustering_multi_runs(embeddings, n_runs=1)
            results.update(clustering_results)

        # Visualization
        if self.args.visualize:
            logger.info("\n" + "-" * 60)
            logger.info("Generating Visualizations")
            logger.info("-" * 60)
            self.gnn_encoder.eval()
            with torch.no_grad():
                embeddings = self.gnn_encoder(self.data.x, self.data.edge_index)
            self._visualize_embeddings(embeddings)

        # Save results
        # self._save_evaluation_results(results)

        logger.info("\n" + "=" * 60)
        logger.info("Evaluation completed!")
        logger.info("=" * 60)

        return results

    def _evaluate_classification_frozen(self, embeddings, n_runs=5):
        """
        Evaluate node classification using frozen embeddings + MLP

        Args:
            embeddings: Pre-computed node embeddings (frozen)
            n_runs: Number of evaluation runs

        Returns:
            Dictionary with mean and std of metrics
        """
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.metrics import accuracy_score, f1_score

        X = embeddings.cpu()
        y = self.data.y.cpu()
        train_mask = self.data.train_mask.cpu()
        val_mask = self.data.val_mask.cpu()
        test_mask = self.data.test_mask.cpu()

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

            # Initialize MLP classifier
            mlp = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(128, n_classes)
            )

            # Move to device
            mlp = mlp.to(self.device)
            X_device = X.to(self.device)
            y_device = y.to(self.device)

            # Optimizer
            optimizer = optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()

            # Training
            best_val_acc = 0
            patience_counter = 0
            patience = 50

            for epoch in range(500):
                mlp.train()
                optimizer.zero_grad()

                # Forward
                logits = mlp(X_device)
                loss = criterion(logits[train_mask], y_device[train_mask])

                # Backward
                loss.backward()
                optimizer.step()

                # Validation
                mlp.eval()
                with torch.no_grad():
                    val_logits = mlp(X_device)
                    val_pred = val_logits[val_mask].argmax(dim=1)
                    val_acc = accuracy_score(
                        y_device[val_mask].cpu().numpy(),
                        val_pred.cpu().numpy()
                    )

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state = mlp.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Load best model
            mlp.load_state_dict(best_model_state)

            # Final evaluation
            mlp.eval()
            with torch.no_grad():
                logits = mlp(X_device)

                # Validation metrics
                val_pred = logits[val_mask].argmax(dim=1).cpu().numpy()
                val_true = y_device[val_mask].cpu().numpy()
                val_acc = accuracy_score(val_true, val_pred)

                # Test metrics
                test_pred = logits[test_mask].argmax(dim=1).cpu().numpy()
                test_true = y_device[test_mask].cpu().numpy()
                test_acc = accuracy_score(test_true, test_pred)
                test_f1_micro = f1_score(test_true, test_pred, average='micro')
                test_f1_macro = f1_score(test_true, test_pred, average='macro')

            # Store results
            all_results['val_acc'].append(val_acc)
            all_results['test_acc'].append(test_acc)
            all_results['test_f1_micro'].append(test_f1_micro)
            all_results['test_f1_macro'].append(test_f1_macro)

            logger.info(f"    Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, "
                        f"F1-Micro: {test_f1_micro:.4f}, F1-Macro: {test_f1_macro:.4f}")

        # Compute statistics
        results = {}
        for metric, values in all_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            results[f'{metric}_frozen_mean'] = mean_val
            results[f'{metric}_frozen_std'] = std_val

        # Log summary
        logger.info("\n  Classification Summary (Frozen Embeddings):")
        logger.info(f"    Val Accuracy:    {results['val_acc_frozen_mean']:.4f} ± {results['val_acc_frozen_std']:.4f}")
        logger.info(
            f"    Test Accuracy:   {results['test_acc_frozen_mean']:.4f} ± {results['test_acc_frozen_std']:.4f}")
        logger.info(
            f"    Test F1-Micro:   {results['test_f1_micro_frozen_mean']:.4f} ± {results['test_f1_micro_frozen_std']:.4f}")
        logger.info(
            f"    Test F1-Macro:   {results['test_f1_macro_frozen_mean']:.4f} ± {results['test_f1_macro_frozen_std']:.4f}")

        return results

    def _evaluate_classification_finetune(self, n_runs=5):
        """
        Evaluate node classification with full fine-tuning of GNN backbone

        Args:
            n_runs: Number of evaluation runs

        Returns:
            Dictionary with mean and std of metrics
        """
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.metrics import accuracy_score, f1_score
        import copy

        y = self.data.y
        train_mask = self.data.train_mask
        val_mask = self.data.val_mask
        test_mask = self.data.test_mask

        n_classes = len(torch.unique(y))

        all_results = {
            'val_acc': [],
            'test_acc': [],
            'test_f1_micro': [],
            'test_f1_macro': []
        }

        # Save original GNN state
        original_gnn_state = copy.deepcopy(self.gnn_encoder.state_dict())

        for run in range(n_runs):
            logger.info(f"  Run {run + 1}/{n_runs}")

            # Reset GNN to original state
            self.gnn_encoder.load_state_dict(original_gnn_state)

            # Create classification head
            classifier = nn.Linear(self.args.output_dim, n_classes).to(self.device)

            # Optimizer for both GNN and classifier
            optimizer = optim.Adam(
                list(self.gnn_encoder.parameters()) + list(classifier.parameters()),
                lr=self.args.finetune_lr if hasattr(self.args, 'finetune_lr') else 0.001,
                weight_decay=self.args.finetune_weight_decay if hasattr(self.args, 'finetune_weight_decay') else 5e-4
            )

            criterion = nn.CrossEntropyLoss()

            # Fine-tuning
            best_val_acc = 0
            patience_counter = 0
            patience = 10

            for epoch in range(100):
                # Training
                self.gnn_encoder.train()
                classifier.train()
                optimizer.zero_grad()

                # Forward
                embeddings = self.gnn_encoder(self.data.x, self.data.edge_index)
                logits = classifier(embeddings)
                loss = criterion(logits[train_mask], y[train_mask])

                # Backward
                loss.backward()
                optimizer.step()

                # Validation
                self.gnn_encoder.eval()
                classifier.eval()
                with torch.no_grad():
                    val_embeddings = self.gnn_encoder(self.data.x, self.data.edge_index)
                    val_logits = classifier(val_embeddings)
                    val_pred = val_logits[val_mask].argmax(dim=1)
                    val_acc = accuracy_score(
                        y[val_mask].cpu().numpy(),
                        val_pred.cpu().numpy()
                    )

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_gnn_state = copy.deepcopy(self.gnn_encoder.state_dict())
                    best_classifier_state = copy.deepcopy(classifier.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Load best model
            self.gnn_encoder.load_state_dict(best_gnn_state)
            classifier.load_state_dict(best_classifier_state)

            # Final evaluation
            self.gnn_encoder.eval()
            classifier.eval()
            with torch.no_grad():
                embeddings = self.gnn_encoder(self.data.x, self.data.edge_index)
                logits = classifier(embeddings)

                # Validation metrics
                val_pred = logits[val_mask].argmax(dim=1).cpu().numpy()
                val_true = y[val_mask].cpu().numpy()
                val_acc = accuracy_score(val_true, val_pred)

                # Test metrics
                test_pred = logits[test_mask].argmax(dim=1).cpu().numpy()
                test_true = y[test_mask].cpu().numpy()
                test_acc = accuracy_score(test_true, test_pred)
                test_f1_micro = f1_score(test_true, test_pred, average='micro')
                test_f1_macro = f1_score(test_true, test_pred, average='macro')

            # Store results
            all_results['val_acc'].append(val_acc)
            all_results['test_acc'].append(test_acc)
            all_results['test_f1_micro'].append(test_f1_micro)
            all_results['test_f1_macro'].append(test_f1_macro)

            logger.info(f"    Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, "
                        f"F1-Micro: {test_f1_micro:.4f}, F1-Macro: {test_f1_macro:.4f}")

        self.gnn_encoder.eval()

        # Restore original GNN state
        self.gnn_encoder.load_state_dict(original_gnn_state)

        # Compute statistics
        results = {}
        for metric, values in all_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            results[f'{metric}_finetune_mean'] = mean_val
            results[f'{metric}_finetune_std'] = std_val

        # Log summary
        logger.info("\n  Classification Summary (Fine-tuned GNN):")
        logger.info(
            f"    Val Accuracy:    {results['val_acc_finetune_mean']:.4f} ± {results['val_acc_finetune_std']:.4f}")
        logger.info(
            f"    Test Accuracy:   {results['test_acc_finetune_mean']:.4f} ± {results['test_acc_finetune_std']:.4f}")
        logger.info(
            f"    Test F1-Micro:   {results['test_f1_micro_finetune_mean']:.4f} ± {results['test_f1_micro_finetune_std']:.4f}")
        logger.info(
            f"    Test F1-Macro:   {results['test_f1_macro_finetune_mean']:.4f} ± {results['test_f1_macro_finetune_std']:.4f}")

        return results

    def _evaluate_clustering_multi_runs(self, embeddings, n_runs=5):
        """
        Evaluate clustering with multiple runs
        Reports mean and std of metrics

        Args:
            embeddings: Node embeddings
            n_runs: Number of evaluation runs

        Returns:
            Dictionary with mean and std of metrics
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

        X = embeddings.cpu().numpy()
        X = normalize(X, norm='l2')
        y_true = self.data.y.cpu().numpy()
        n_clusters = len(np.unique(y_true))

        all_results = {
            'nmi': [],
            'ari': []
        }

        for run in range(n_runs):
            # KMeans with different random states
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42 + run,
                n_init=10,
                max_iter=300
            )
            y_pred = kmeans.fit_predict(X)

            # Compute metrics
            nmi = normalized_mutual_info_score(y_true, y_pred)
            ari = adjusted_rand_score(y_true, y_pred)

            all_results['nmi'].append(nmi)
            all_results['ari'].append(ari)

            logger.info(f"  Run {run + 1}/{n_runs}: NMI: {nmi:.4f}, ARI: {ari:.4f}")

        # Compute statistics
        results = {}
        for metric, values in all_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            results[f'{metric}_mean'] = mean_val
            results[f'{metric}_std'] = std_val

        # Log summary
        logger.info("\n  Clustering Summary:")
        logger.info(f"    NMI: {results['nmi_mean']:.4f} ± {results['nmi_std']:.4f}")
        logger.info(f"    ARI: {results['ari_mean']:.4f} ± {results['ari_std']:.4f}")

        return results

    def _save_evaluation_results(self, results):
        """Save evaluation results to file"""
        import json
        from pathlib import Path

        output_dir = Path(self.args.output_dir) / self.args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / 'evaluation_results.json'

        # Convert numpy types to Python types for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, (np.floating, np.integer)):
                results_serializable[key] = float(value)
            else:
                results_serializable[key] = value

        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"\nResults saved to {results_path}")

    def _visualize_embeddings(self, embeddings):
        """Visualize embeddings using t-SNE"""
        logger.info("Generating t-SNE visualization...")

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        X = embeddings.cpu().numpy()

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_2d = tsne.fit_transform(X)

        # Plot
        plt.figure(figsize=(10, 8))

        if hasattr(self.data, 'y'):
            y = self.data.y.cpu().numpy()
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter)
        else:
            plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6)

        plt.title('t-SNE Visualization of Learned Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')

        # Save
        output_dir = Path(self.args.output_dir) / self.args.dataset
        plt.savefig(output_dir / 'tsne_visualization.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved to {output_dir / 'tsne_visualization.pdf'}")
