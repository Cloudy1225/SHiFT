"""
taxonomy_update.py

Incremental taxonomy update when document embeddings change
Uses mapping & diagnosis approach to minimize LLM calls
"""
import copy
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set

import json_repair
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy import ConceptNode, TaxonomyVisualizer, TaxonomyQuery
from taxonomy_initialize import LLMClient, CoreSetSampler

logger = logging.getLogger(__name__)


@dataclass
class NodeUpdateMetadata:
    """Metadata for tracking node changes during update"""
    node_id: int  # Use id(node) as unique identifier
    old_variance: float = 0.0
    new_variance: float = 0.0
    old_size: int = 0
    new_size: int = 0
    old_doc_set: Set[int] = None
    new_doc_set: Set[int] = None
    unassigned_docs: List[int] = None

    def __post_init__(self):
        if self.old_doc_set is None:
            self.old_doc_set = set()
        if self.new_doc_set is None:
            self.new_doc_set = set()
        if self.unassigned_docs is None:
            self.unassigned_docs = []

    @property
    def variance_ratio(self) -> float:
        """Calculate variance increase ratio"""
        return self.new_variance / (self.old_variance + 1e-10)

    @property
    def size_ratio(self) -> float:
        """Calculate size increase ratio"""
        return self.new_size / (self.old_size + 1e-10)

    @property
    def turnover_rate(self) -> float:
        """Calculate document turnover rate"""
        if not self.old_doc_set:
            return 0.0
        retained = self.old_doc_set & self.new_doc_set
        return 1.0 - (len(retained) / len(self.old_doc_set))


class TaxonomyUpdater:
    """Incrementally update taxonomy when embeddings change"""

    def __init__(
            self,
            llm_client: LLMClient,
            sampler: CoreSetSampler,
            cosine_threshold: float = 0.7,
            merge_threshold: float = 0.9,
            variance_increase_ratio: float = 1.5,
            size_increase_ratio: float = 2.5,
            turnover_threshold: float = 0.5,
            min_cluster_size: int = 5,
            max_depth: int = 3,
            min_docs_for_split: int = 50,
            llm_max_retries: int = 3
    ):
        """
        Initialize taxonomy updater

        Args:
            llm_client: LLM client for API calls
            sampler: Core-set sampler for representative documents
            cosine_threshold: Minimum cosine similarity for assignment (default: 0.7)
            merge_threshold: Cosine similarity threshold for merging clusters (default: 0.9)
            variance_increase_ratio: Ratio threshold for detecting cluster split (default: 1.5)
            size_increase_ratio: Size ratio threshold for detecting cluster split (default: 2.5)
            turnover_threshold: Document turnover ratio for definition drift (default: 0.5)
            min_cluster_size: Minimum cluster size before pruning (default: 5)
            max_depth: Maximum tree depth (default: 3)
            min_docs_for_split: Minimum documents required for splitting (default: 50)
            llm_max_retries: Maximum retries for LLM calls (default: 3)
        """
        self.llm_client = llm_client
        self.sampler = sampler
        self.cosine_threshold = cosine_threshold
        self.merge_threshold = merge_threshold
        self.variance_increase_ratio = variance_increase_ratio
        self.size_increase_ratio = size_increase_ratio
        self.turnover_threshold = turnover_threshold
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth
        self.min_docs_for_split = min_docs_for_split
        self.llm_max_retries = llm_max_retries

        # Update metadata storage (node_id -> NodeUpdateMetadata)
        self.node_metadata: Dict[int, NodeUpdateMetadata] = {}

        # Track old assignments for drift detection
        self.old_assignments: Dict[int, str] = {}

        # Statistics
        self.stats = {
            'splits': 0,
            'merges': 0,
            'drift_corrections': 0,
            'pruned': 0,
            'outlier_reassignments': 0,
            'llm_calls': 0
        }

        logger.info("Initialized TaxonomyUpdater with parameters:")
        logger.info(f"  Cosine threshold: {cosine_threshold}")
        logger.info(f"  Merge threshold: {merge_threshold}")
        logger.info(f"  Variance increase ratio: {variance_increase_ratio}")
        logger.info(f"  Size increase ratio: {size_increase_ratio}")
        logger.info(f"  Turnover threshold: {turnover_threshold}")
        logger.info(f"  Min cluster size: {min_cluster_size}")

    def update_taxonomy(
            self,
            root: ConceptNode,
            old_embeddings: np.ndarray,
            new_embeddings: np.ndarray,
            raw_texts: List[str],
            edge_index: np.ndarray
    ) -> ConceptNode:
        """
        Update taxonomy with new embeddings

        Args:
            root: Root node of existing taxonomy
            old_embeddings: Original embeddings used to build taxonomy
            new_embeddings: Updated embeddings
            raw_texts: Raw text content of documents
            edge_index: Graph edge index

        Returns:
            Updated root node
        """
        logger.info("=" * 60)
        logger.info("Starting Incremental Taxonomy Update")
        logger.info("=" * 60)

        root = copy.deepcopy(root)

        # Reset state
        self.node_metadata = {}
        self.old_assignments = {}
        self.stats = {k: 0 for k in self.stats}

        # Step 1: Centroid Re-alignment
        logger.info("Step 1: Centroid Re-alignment")
        self.old_assignments = self._get_current_assignments(root)
        self._realign_centroids(root, old_embeddings, new_embeddings)

        # Step 2: Global Re-assignment
        logger.info("Step 2: Global Re-assignment")
        self._global_reassignment(root, new_embeddings)

        # Step 3: Surgical LLM Intervention
        logger.info("Step 3: Surgical LLM Intervention")
        self._surgical_intervention(
            root, old_embeddings, new_embeddings,
            raw_texts, edge_index
        )

        # Final: Update all centroids
        logger.info("Final: Updating all centroids")
        self._update_all_centroids(root, new_embeddings)

        # Cleanup metadata
        self.node_metadata.clear()
        self.old_assignments.clear()

        # Print statistics
        self._print_statistics()

        return root

    def _get_or_create_metadata(self, node: ConceptNode) -> NodeUpdateMetadata:
        """Get or create metadata for a node"""
        node_id = id(node)
        if node_id not in self.node_metadata:
            self.node_metadata[node_id] = NodeUpdateMetadata(node_id=node_id)
        return self.node_metadata[node_id]

    def _realign_centroids(
            self,
            node: ConceptNode,
            old_embeddings: np.ndarray,
            new_embeddings: np.ndarray
    ):
        """
        Step 1: Update centroids using new embeddings while keeping membership unchanged
        """
        if not node.doc_indices:
            return

        metadata = self._get_or_create_metadata(node)

        # Store old statistics
        if old_embeddings is not None and len(node.doc_indices) > 0:
            old_docs_embeddings = old_embeddings[node.doc_indices]
            old_centroid = old_docs_embeddings.mean(axis=0)
            old_variance = np.var(np.linalg.norm(
                old_docs_embeddings - old_centroid, axis=1
            ))

            metadata.old_variance = old_variance
            metadata.old_size = len(node.doc_indices)
            metadata.old_doc_set = set(node.doc_indices)

        # Update centroid with new embeddings
        new_docs_embeddings = new_embeddings[node.doc_indices]
        node.prototype = new_docs_embeddings.mean(axis=0)

        # Calculate new variance
        new_variance = np.var(np.linalg.norm(
            new_docs_embeddings - node.prototype, axis=1
        ))
        metadata.new_variance = new_variance

        # Recursively update children
        for child in node.children:
            self._realign_centroids(child, old_embeddings, new_embeddings)

    def _get_current_assignments(self, root: ConceptNode) -> Dict[int, str]:
        """Get current document-to-leaf assignments"""
        assignments = {}

        def traverse(node: ConceptNode, path: List[str]):
            current_path = path + [node.name]

            if not node.children:  # Leaf node
                path_str = " > ".join(current_path)
                for doc_idx in node.doc_indices:
                    assignments[doc_idx] = path_str
            else:
                for child in node.children:
                    traverse(child, current_path)

        traverse(root, [])
        return assignments

    def _global_reassignment(
            self,
            root: ConceptNode,
            new_embeddings: np.ndarray
    ):
        """
        Step 2: Reassign all documents using top-down greedy assignment
        """
        # Clear all existing document assignments in the tree
        self._clear_assignments(root)

        # Assign each document top-down (pushes docs to leaves or unassigned buckets)
        n_docs = len(new_embeddings)
        for doc_idx in range(n_docs):
            doc_embedding = new_embeddings[doc_idx]
            self._assign_document_topdown(root, doc_idx, doc_embedding)

        # Aggregate indices bottom-up
        # This repopulates parent nodes with their children's documents
        self._aggregate_indices_bottom_up(root)

        # Update metadata stats (size, variance) based on the fully aggregated lists
        self._update_new_metadata(root)

    def _aggregate_indices_bottom_up(self, node: ConceptNode) -> Set[int]:
        """
        Helper: Recursively aggregates document indices from children to parents.

        This ensures that intermediate (parent) nodes contain the union of all
        documents in their subtree, which is critical for calculating accurate
        turnover rates and variance statistics.
        """
        # 1. Start with documents directly assigned to this node
        # (Usually populated for leaf nodes, or outliers assigned to this specific level)
        all_indices = set(node.doc_indices)

        # 2. Include 'unassigned' documents currently stored in metadata
        # These are docs that reached this node but didn't fit any child
        metadata = self.node_metadata.get(id(node))
        if metadata and metadata.unassigned_docs:
            all_indices.update(metadata.unassigned_docs)

        # 3. Recursively collect indices from all children
        for child in node.children:
            child_indices = self._aggregate_indices_bottom_up(child)
            all_indices.update(child_indices)

        # 4. Update the node's official document list with the aggregated set
        # This fixes the "empty parent" bug causing 100% turnover
        node.doc_indices = list(all_indices)

        return all_indices

    def _clear_assignments(self, node: ConceptNode):
        """Recursively clear all document assignments"""
        node.doc_indices = []

        # Clear unassigned list in metadata
        metadata = self._get_or_create_metadata(node)
        metadata.unassigned_docs = []

        for child in node.children:
            self._clear_assignments(child)

    def _assign_document_topdown(
            self,
            node: ConceptNode,
            doc_idx: int,
            doc_embedding: np.ndarray
    ):
        """
        Assign a document top-down from current node
        """
        # If leaf node, assign here
        if not node.children:
            node.doc_indices.append(doc_idx)
            return

        # Find best matching child
        best_child = None
        best_similarity = -1.0

        for child in node.children:
            if child.prototype is not None:
                similarity = cosine_similarity(
                    doc_embedding.reshape(1, -1),
                    child.prototype.reshape(1, -1)
                )[0, 0]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_child = child

        # Assign to best child or mark as unassigned
        if best_similarity >= self.cosine_threshold and best_child is not None:
            self._assign_document_topdown(best_child, doc_idx, doc_embedding)
        else:
            # Add to unassigned bucket at this level
            metadata = self._get_or_create_metadata(node)
            metadata.unassigned_docs.append(doc_idx)
            logger.debug(f"Document {doc_idx} marked as unassigned at node '{node.name}'")

    def _update_new_metadata(self, node: ConceptNode):
        """Update new_size and new_doc_set after reassignment"""
        metadata = self._get_or_create_metadata(node)
        metadata.new_size = len(node.doc_indices)
        metadata.new_doc_set = set(node.doc_indices)

        for child in node.children:
            self._update_new_metadata(child)

    def _surgical_intervention(
            self,
            root: ConceptNode,
            old_embeddings: np.ndarray,
            new_embeddings: np.ndarray,
            raw_texts: List[str],
            edge_index: np.ndarray
    ):
        """
        Step 3: Apply surgical LLM interventions where needed
        """
        self._process_node_interventions(
            root, old_embeddings, new_embeddings,
            raw_texts, edge_index
        )

    def _process_node_interventions(
            self,
            node: ConceptNode,
            old_embeddings: np.ndarray,
            new_embeddings: np.ndarray,
            raw_texts: List[str],
            edge_index: np.ndarray
    ):
        """
        Process interventions for current node and its children
        """
        # Process children first (bottom-up)
        children_to_remove = []

        for child in node.children:
            self._process_node_interventions(
                child, old_embeddings, new_embeddings,
                raw_texts, edge_index
            )

            # Check if child should be pruned after processing
            if self._should_prune(child):
                children_to_remove.append(child)

        # D: Automatic Pruning
        for child in children_to_remove:
            self._prune_node(node, child, new_embeddings)

        # Only process non-root nodes with children
        if node.children and node.depth > 0:
            # B: Concept Merging
            self._check_and_merge_siblings(node, new_embeddings, raw_texts, edge_index)

        # Process current node's issues
        if node.depth > 0:  # Skip root
            # A: Concept Splitting
            if self._should_split(node):
                self._split_node(node, new_embeddings, raw_texts, edge_index)

            # C: Definition Drift
            elif self._has_definition_drift(node):
                self._correct_definition_drift(node, new_embeddings, raw_texts, edge_index)

        # E: Outlier Reassignment
        metadata = self._get_or_create_metadata(node)
        if metadata.unassigned_docs:
            self._reassign_outliers(node, new_embeddings, raw_texts)

    def _should_split(self, node: ConceptNode) -> bool:
        """Check if node should be split"""
        # Need enough documents
        if len(node.doc_indices) < self.min_docs_for_split:
            return False

        # Check if already at max depth
        if node.depth >= self.max_depth:
            return False

        metadata = self._get_or_create_metadata(node)

        # Check variance increase
        if metadata.old_variance > 0:
            variance_ratio = metadata.variance_ratio
            if variance_ratio > self.variance_increase_ratio:
                logger.info(f"Node '{node.name}' variance increased {variance_ratio:.2f}x")
                return True

        # Check size increase
        if metadata.old_size > 0:
            size_ratio = metadata.size_ratio
            if size_ratio > self.size_increase_ratio:
                logger.info(f"Node '{node.name}' size increased {size_ratio:.2f}x")
                return True

        return False

    def _split_node(
            self,
            node: ConceptNode,
            new_embeddings: np.ndarray,
            raw_texts: List[str],
            edge_index: np.ndarray
    ):
        """
        A: Split a node into multiple sub-concepts
        """
        logger.info(f"Splitting node '{node.name}' with {len(node.doc_indices)} documents")

        # Get embeddings for this cluster
        cluster_embeddings = new_embeddings[node.doc_indices]
        normalized_embeddings = normalize(cluster_embeddings, norm='l2')

        # Determine number of splits (2-4 clusters)
        n_clusters = min(4, max(2, int(np.sqrt(len(node.doc_indices) / 50))))

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_embeddings)
        centers = kmeans.cluster_centers_

        # Organize documents by cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(node.doc_indices[i])

        # Generate concepts for each cluster
        new_children = []
        for cluster_id, cluster_docs in enumerate(clusters):
            if len(cluster_docs) == 0:
                continue

            # Sample representative documents
            sampled_indices = self.sampler.sample(
                embeddings=new_embeddings,
                doc_indices=cluster_docs,
                cluster_center=centers[cluster_id],
                edge_index=edge_index
            )

            # Generate concept via LLM
            concept_data = self._llm_generate_concept(
                sampled_indices, raw_texts, node
            )

            # Create child node
            child = ConceptNode(
                name=concept_data['name'],
                definition=concept_data['definition'],
                doc_indices=cluster_docs,
                parent=node,
                depth=node.depth + 1,
                split_needed=concept_data.get('split_needed', True),
                prototype=new_embeddings[cluster_docs].mean(axis=0)
            )

            new_children.append(child)
            logger.info(f"  Created sub-concept '{child.name}' with {len(cluster_docs)} documents")

        # Replace node's children with new splits
        node.children = new_children
        self.stats['splits'] += 1
        self.stats['llm_calls'] += len(new_children)

    def _should_prune(self, node: ConceptNode) -> bool:
        """Check if node should be pruned due to too few documents"""
        return len(node.doc_indices) < self.min_cluster_size

    def _prune_node(
            self,
            parent: ConceptNode,
            node: ConceptNode,
            new_embeddings: np.ndarray
    ):
        """
        D: Prune a node and mark its documents as unassigned
        """
        logger.info(f"Pruning node '{node.name}' with only {len(node.doc_indices)} documents")

        # Add documents to parent's unassigned bucket
        parent_metadata = self._get_or_create_metadata(parent)
        parent_metadata.unassigned_docs.extend(node.doc_indices)

        # Remove node from parent's children
        parent.children.remove(node)

        # Clean up metadata for pruned node
        node_id = id(node)
        if node_id in self.node_metadata:
            del self.node_metadata[node_id]

        self.stats['pruned'] += 1

    def _check_and_merge_siblings(
            self,
            parent: ConceptNode,
            new_embeddings: np.ndarray,
            raw_texts: List[str],
            edge_index: np.ndarray
    ):
        """
        B: Check for and merge similar sibling concepts (supports multi-way merging)
        """
        children = parent.children
        if len(children) < 2:
            return

        # Build similarity matrix for all siblings
        n_children = len(children)
        similarity_matrix = np.zeros((n_children, n_children))

        for i in range(n_children):
            for j in range(i + 1, n_children):
                if children[i].prototype is None or children[j].prototype is None:
                    continue

                sim = cosine_similarity(
                    children[i].prototype.reshape(1, -1),
                    children[j].prototype.reshape(1, -1)
                )[0, 0]
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # Find groups of similar concepts using connected components
        merge_groups = self._find_merge_groups(
            children, similarity_matrix, self.merge_threshold
        )

        if not merge_groups:
            logger.info(f"No merge candidates found under '{parent.name}'")
            return

        logger.info(f"Found {len(merge_groups)} potential merge groups under '{parent.name}'")

        # Process each group with LLM
        merged_nodes = []
        indices_to_remove = set()

        for group_indices in merge_groups:
            if len(group_indices) < 2:
                continue

            group_children = [children[i] for i in group_indices]

            logger.info(f"Checking merge for {len(group_children)} concepts: "
                        f"{[c.name for c in group_children]}")

            # Ask LLM about this group
            merge_plan = self._llm_check_merge_group(
                group_children, new_embeddings, raw_texts, parent
            )

            if merge_plan and merge_plan['operations']:
                # Execute merge operations
                for op in merge_plan['operations']:
                    if op['type'] == 'MERGE':
                        # Get indices relative to the group
                        local_indices = op['source_indices']
                        global_indices = [group_indices[i] for i in local_indices]

                        # Merge these concepts
                        merged_node = self._execute_merge(
                            [children[i] for i in global_indices],
                            op['merged_name'],
                            op['merged_definition'],
                            parent,
                            new_embeddings
                        )

                        merged_nodes.append(merged_node)
                        indices_to_remove.update(global_indices)

                        logger.info(f"Merged {len(global_indices)} concepts into "
                                    f"'{merged_node.name}'")
                        self.stats['merges'] += 1

                    elif op['type'] == 'KEEP_SEPARATE':
                        # Do nothing, concepts remain separate
                        kept_names = [group_children[i].name for i in op['indices']]
                        logger.info(f"Keeping separate: {kept_names}")

                self.stats['llm_calls'] += 1

        # Remove merged nodes and add new merged nodes
        if indices_to_remove:
            # Clean up metadata for removed nodes
            for idx in indices_to_remove:
                node_id = id(children[idx])
                if node_id in self.node_metadata:
                    del self.node_metadata[node_id]

            # Remove old nodes
            parent.children = [
                child for i, child in enumerate(children)
                if i not in indices_to_remove
            ]

            # Add merged nodes
            parent.children.extend(merged_nodes)

            logger.info(f"Completed merging: removed {len(indices_to_remove)} nodes, "
                        f"added {len(merged_nodes)} merged nodes")

    def _find_merge_groups(
            self,
            children: List[ConceptNode],
            similarity_matrix: np.ndarray,
            threshold: float
    ) -> List[List[int]]:
        """
        Find groups of mutually similar concepts using graph connectivity

        Args:
            children: List of child nodes
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold for grouping

        Returns:
            List of groups, where each group is a list of indices
        """
        n = len(children)

        # Build adjacency list for similarity graph
        adj_list = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        # Find connected components using DFS
        visited = set()
        groups = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for i in range(n):
            if i not in visited and i in adj_list:  # Only nodes with connections
                component = []
                dfs(i, component)
                if len(component) >= 2:  # Only groups with 2+ members
                    groups.append(sorted(component))

        return groups

    def _llm_check_merge_group(
            self,
            group_children: List[ConceptNode],
            embeddings: np.ndarray,
            raw_texts: List[str],
            parent: ConceptNode
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a group of similar concepts should be merged (supports multi-way merge)

        Args:
            group_children: List of similar child concepts
            embeddings: Document embeddings
            raw_texts: Raw text content
            parent: Parent node

        Returns:
            Dictionary with merge plan, or None if parsing fails
        """
        # Prepare concept information
        concepts_info = []
        for i, child in enumerate(group_children):
            # Get average similarity to others in group
            avg_sim = 0
            if len(group_children) > 1:
                sims = []
                for j, other in enumerate(group_children):
                    if i != j and child.prototype is not None and other.prototype is not None:
                        sim = cosine_similarity(
                            child.prototype.reshape(1, -1),
                            other.prototype.reshape(1, -1)
                        )[0, 0]
                        sims.append(sim)
                avg_sim = np.mean(sims) if sims else 0

            concepts_info.append(
                f"{i}. {child.name}\n"
                f"   Definition: {child.definition}\n"
                f"   Documents: {len(child.doc_indices)}\n"
                f"   Avg similarity to group: {avg_sim:.3f}"
            )

        concepts_str = "\n".join(concepts_info)

        system_prompt = "You are an expert taxonomist optimizing a document taxonomy."

        parent_info = f'"{parent.name}"' if parent.depth > 0 else '"Root"'

        user_prompt = f"""Context: Under the parent category {parent_info}, we identified {len(group_children)} sibling concepts that appear semantically similar based on their vector representations.

Similar Concepts:
{concepts_str}

Task: Analyze these concepts and determine the best merging strategy.

You can output ONE of the following strategies:

Strategy A: MERGE ALL
- Merge all {len(group_children)} concepts into a single broader concept
- Use when they are all synonyms or very closely related

Strategy B: PARTIAL MERGE
- Merge some concepts together, keep others separate
- Use when there are subgroups of similar concepts

Strategy C: KEEP ALL SEPARATE
- Don't merge any concepts
- Use when they are genuinely distinct despite high similarity

Consider:
1. Are they synonyms? (e.g., "GNN", "Graph Neural Nets", "Graph NN")
2. Are they closely related subtopics? (e.g., "LSTM", "GRU", "Recurrent Nets")
3. Are they distinct but related? (e.g., "Image Classification", "Object Detection")

Output Format (JSON):
{{
  "strategy": "MERGE_ALL" or "PARTIAL_MERGE" or "KEEP_SEPARATE",
  "reasoning": "Brief explanation of your decision",
  "operations": [
    {{
      "type": "MERGE",
      "source_indices": [0, 2, 3],
      "merged_name": "New unified name",
      "merged_definition": "New unified definition"
    }},
    {{
      "type": "KEEP_SEPARATE",
      "indices": [1, 4],
      "reason": "Why they should stay separate"
    }}
  ]
}}

Examples:

Example 1 - MERGE ALL (synonyms):
{{
  "strategy": "MERGE_ALL",
  "reasoning": "All three are synonyms for the same concept",
  "operations": [
    {{
      "type": "MERGE",
      "source_indices": [0, 1, 2],
      "merged_name": "Graph Neural Networks",
      "merged_definition": "Neural networks designed to operate on graph-structured data"
    }}
  ]
}}

Example 2 - PARTIAL MERGE (subgroups):
{{
  "strategy": "PARTIAL_MERGE",
  "reasoning": "Indices 0,1 are about RNNs, indices 2,3 are about CNNs, index 4 is distinct",
  "operations": [
    {{
      "type": "MERGE",
      "source_indices": [0, 1],
      "merged_name": "Recurrent Neural Networks",
      "merged_definition": "Neural networks with recurrent connections for sequential data"
    }},
    {{
      "type": "MERGE",
      "source_indices": [2, 3],
      "merged_name": "Convolutional Neural Networks",
      "merged_definition": "Neural networks using convolutional layers for spatial data"
    }},
    {{
      "type": "KEEP_SEPARATE",
      "indices": [4],
      "reason": "Transformer architecture is fundamentally different"
    }}
  ]
}}

Example 3 - KEEP SEPARATE (distinct concepts):
{{
  "strategy": "KEEP_SEPARATE",
  "reasoning": "Despite high similarity, these are distinct research directions",
  "operations": [
    {{
      "type": "KEEP_SEPARATE",
      "indices": [0, 1, 2],
      "reason": "Each represents a different approach to the same problem"
    }}
  ]
}}

Now analyze the {len(group_children)} concepts above and output your merge plan:"""

        try:
            result = self._retry_llm_and_parse_json(system_prompt, user_prompt)

            # Validate the response
            if 'operations' not in result:
                logger.warning("LLM response missing 'operations' field")
                return None

            # Validate indices
            n = len(group_children)
            for op in result['operations']:
                if op['type'] == 'MERGE':
                    indices = op.get('source_indices', [])
                    if not indices or any(i < 0 or i >= n for i in indices):
                        logger.warning(f"Invalid merge indices: {indices}")
                        return None
                elif op['type'] == 'KEEP_SEPARATE':
                    indices = op.get('indices', [])
                    if any(i < 0 or i >= n for i in indices):
                        logger.warning(f"Invalid keep indices: {indices}")
                        return None

            return result

        except Exception as e:
            logger.error(f"Failed to check merge group: {e}")
            return None

    def _execute_merge(
            self,
            nodes_to_merge: List[ConceptNode],
            merged_name: str,
            merged_definition: str,
            parent: ConceptNode,
            embeddings: np.ndarray
    ) -> ConceptNode:
        """
        Execute a merge operation for multiple nodes

        Args:
            nodes_to_merge: List of nodes to merge
            merged_name: Name for merged concept
            merged_definition: Definition for merged concept
            parent: Parent node
            embeddings: Document embeddings

        Returns:
            Newly created merged node
        """
        # Collect all documents
        merged_docs = []
        for node in nodes_to_merge:
            merged_docs.extend(node.doc_indices)

        # Remove duplicates and sort
        merged_docs = sorted(list(set(merged_docs)))

        # Create merged node
        merged_node = ConceptNode(
            name=merged_name,
            definition=merged_definition,
            doc_indices=merged_docs,
            parent=parent,
            depth=nodes_to_merge[0].depth,
            split_needed=True,
            prototype=embeddings[merged_docs].mean(axis=0) if merged_docs else None
        )

        # Merge children from all nodes
        all_children = []
        for node in nodes_to_merge:
            all_children.extend(node.children)

        # Update parent references
        for child in all_children:
            child.parent = merged_node

        merged_node.children = all_children

        logger.info(f"Created merged node '{merged_name}' with {len(merged_docs)} documents "
                    f"and {len(all_children)} children")

        return merged_node

    def _has_definition_drift(self, node: ConceptNode) -> bool:
        """
        Check if node has significant definition drift based on document turnover
        """
        metadata = self._get_or_create_metadata(node)

        if metadata.old_size == 0:
            return False

        turnover_rate = metadata.turnover_rate

        if turnover_rate > self.turnover_threshold:
            logger.info(f"Node '{node.name}' has {turnover_rate:.1%} document turnover")
            return True

        return False

    def _get_node_path(self, node: ConceptNode) -> str:
        """Get full path string for a node"""
        path = []
        current = node
        while current is not None:
            path.insert(0, current.name)
            current = current.parent
        return " > ".join(path)

    def _correct_definition_drift(
            self,
            node: ConceptNode,
            new_embeddings: np.ndarray,
            raw_texts: List[str],
            edge_index: Optional[np.ndarray] = None
    ):
        """
        C: Correct definition drift by refining concept definition
        """
        logger.info(f"Correcting definition drift for '{node.name}'")

        # Sample current representative documents
        if len(node.doc_indices) == 0:
            return

        sampled_indices = self.sampler.sample(
            embeddings=new_embeddings,
            doc_indices=node.doc_indices,
            cluster_center=new_embeddings[node.doc_indices].mean(axis=0),
            edge_index=edge_index
        )

        # Ask LLM to refine definition
        refined_concept = self._llm_refine_definition(
            node, sampled_indices, raw_texts
        )

        # Update node
        if refined_concept['should_update']:
            logger.info(f"  Old definition: {node.definition}")
            logger.info(f"  New definition: {refined_concept['definition']}")

            node.name = refined_concept['name']
            node.definition = refined_concept['definition']

            self.stats['drift_corrections'] += 1
            self.stats['llm_calls'] += 1

    def _reassign_outliers(
            self,
            node: ConceptNode,
            new_embeddings: np.ndarray,
            raw_texts: List[str]
    ):
        """
        E: Reassign outlier documents to appropriate clusters
        """
        metadata = self._get_or_create_metadata(node)

        if not metadata.unassigned_docs:
            return

        logger.info(f"Reassigning {len(metadata.unassigned_docs)} outliers at node '{node.name}'")

        # If no children, documents stay in this node
        if not node.children:
            node.doc_indices.extend(metadata.unassigned_docs)
            metadata.unassigned_docs = []
            return

        # For each outlier, find best matching cluster
        for doc_idx in metadata.unassigned_docs:
            doc_embedding = new_embeddings[doc_idx]

            # Find top-3 nearest clusters
            similarities = []
            for child in node.children:
                if child.prototype is not None:
                    sim = cosine_similarity(
                        doc_embedding.reshape(1, -1),
                        child.prototype.reshape(1, -1)
                    )[0, 0]
                    similarities.append((sim, child))

            if not similarities:
                continue

            similarities.sort(reverse=True, key=lambda x: x[0])
            top_candidates = similarities[:min(3, len(similarities))]

            # If top candidate is very close, assign directly
            if top_candidates[0][0] > 0.85:
                best_child = top_candidates[0][1]
                best_child.doc_indices.append(doc_idx)
                logger.debug(f"  Document {doc_idx} auto-assigned to '{best_child.name}' "
                             f"(similarity: {top_candidates[0][0]:.3f})")
                self.stats['outlier_reassignments'] += 1
                continue

            # Otherwise, ask LLM
            best_child = self._llm_assign_outlier(
                doc_idx, raw_texts, top_candidates
            )

            if best_child is not None:
                best_child.doc_indices.append(doc_idx)
                logger.debug(f"  Document {doc_idx} LLM-assigned to '{best_child.name}'")
                self.stats['outlier_reassignments'] += 1
                self.stats['llm_calls'] += 1
            else:
                # If LLM can't decide, assign to most similar
                best_child = top_candidates[0][1]
                best_child.doc_indices.append(doc_idx)
                logger.debug(f"  Document {doc_idx} fallback-assigned to '{best_child.name}'")

        # Clear unassigned bucket
        metadata.unassigned_docs = []

    def _update_all_centroids(self, node: ConceptNode, embeddings: np.ndarray):
        """Recursively update all centroids with final document assignments"""
        if node.doc_indices:
            node.prototype = embeddings[node.doc_indices].mean(axis=0)

        for child in node.children:
            self._update_all_centroids(child, embeddings)

    # ==================== LLM Helper Methods ====================

    def _retry_llm_and_parse_json(self, system_prompt: str, user_prompt: str):
        """
        Retry wrapper for LLM call + JSON parsing.
        """
        last_err = None

        for attempt in range(self.llm_max_retries):
            try:
                # Call LLM
                response = self.llm_client.call(system_prompt, user_prompt)

                # Parse JSON
                return json.loads(json_repair.repair_json(response))
            except Exception as e:
                last_err = e
                logger.warning(f"Attempt {attempt + 1}/{self.llm_max_retries} failed in LLM or JSON parsing: {e}")

        logger.error(f"All {self.llm_max_retries} attempts failed. Last response: {response}")
        raise last_err

    def _llm_generate_concept(
            self,
            sampled_indices: List[int],
            raw_texts: List[str],
            parent: ConceptNode
    ) -> Dict[str, Any]:
        """Generate concept using LLM"""
        # Prepare document texts
        doc_texts = []
        for i, idx in enumerate(sampled_indices, 1):
            text = raw_texts[idx][:500]
            doc_texts.append(f"{i}. {text}")

        docs_str = "\n".join(doc_texts)

        system_prompt = "You are an expert taxonomist assisting in organizing scientific literature."

        parent_info = f'"{parent.name}" ({parent.definition})' if parent.depth > 0 else '"All Documents" (Root category)'

        user_prompt = f"""Context: We are currently exploring sub-topics under the parent category: {parent_info}.

Task: Below are {len(sampled_indices)} representative papers from a specific cluster of documents within this parent category.
Your goal is to summarize the common research topic shared by these papers into a specific sub-concept.

Additionally, determine whether this sub-concept is semantically coherent and focused enough that it should NOT be further subdivided.
If the papers clearly belong to a single, narrow topic with no distinct sub-directions, set "concept_split_needed" to false.
If there appear to be multiple distinct sub-themes or methodological directions, set it to true.

Input Papers:
{docs_str}

Constraints:
1. The Concept Name should be concise (2-5 words) and standard in the field.
2. The Definition should be a single sentence explaining what this sub-concept focuses on, distinct from other sub-fields.
3. Ensure the concept strictly belongs to {parent_info}.

Output Format (JSON):
{{
  "concept_name": "String",
  "definition": "String",
  "concept_split_needed": true or false
}}"""

        try:
            concept_data = self._retry_llm_and_parse_json(system_prompt, user_prompt)
            return {
                'name': concept_data.get('concept_name', 'Unknown Concept'),
                'definition': concept_data.get('definition', 'No definition provided'),
                'split_needed': concept_data.get('concept_split_needed', True)
            }
        except Exception as e:
            logger.error(f"Failed to generate concept: {e}")
            return {
                'name': 'Cluster Concept',
                'definition': 'Auto-generated concept (LLM failed)',
                'split_needed': True
            }

    def _llm_refine_definition(
            self,
            node: ConceptNode,
            sampled_indices: List[int],
            raw_texts: List[str]
    ) -> Dict[str, Any]:
        """Refine concept definition based on new documents"""
        # Prepare document texts
        doc_texts = []
        for i, idx in enumerate(sampled_indices, 1):
            text = raw_texts[idx][:500]
            doc_texts.append(f"{i}. {text}")

        docs_str = "\n".join(doc_texts)

        system_prompt = "You are an expert taxonomist refining concept definitions."

        user_prompt = f"""Context: We have an existing concept whose member documents have changed significantly after embedding updates.

Current Concept:
- Name: {node.name}
- Definition: {node.definition}

New Representative Papers:
{docs_str}

Task: Determine if the current definition still accurately describes these papers, or if it needs refinement.

Consider:
1. Does the old definition still capture the main theme of these papers?
2. Has the focus shifted slightly (e.g., from theory to applications)?
3. Should we keep the name but refine the definition?

Output Format (JSON):
{{
  "should_update": true or false,
  "reason": "Brief explanation",
  "new_name": "Refined name (can be same as old)",
  "new_definition": "Refined definition"
}}"""

        try:
            result = self._retry_llm_and_parse_json(system_prompt, user_prompt)

            return {
                'should_update': result.get('should_update', False),
                'name': result.get('new_name', node.name),
                'definition': result.get('new_definition', node.definition),
                'reason': result.get('reason', '')
            }

        except Exception as e:
            logger.error(f"Failed to refine definition: {e}")
            return {
                'should_update': False,
                'name': node.name,
                'definition': node.definition,
                'reason': 'LLM call failed'
            }

    def _llm_assign_outlier(
            self,
            doc_idx: int,
            raw_texts: List[str],
            top_candidates: List[Tuple[float, ConceptNode]]
    ) -> Optional[ConceptNode]:
        """Use LLM to assign outlier document to best cluster"""
        # Prepare document text
        doc_text = raw_texts[doc_idx][:500]

        # Prepare candidate concepts
        candidates_str = []
        for i, (similarity, child) in enumerate(top_candidates, 1):
            candidates_str.append(
                f"{i}. {child.name} (similarity: {similarity:.3f})\n"
                f"   Definition: {child.definition}"
            )

        candidates_text = "\n".join(candidates_str)

        system_prompt = "You are an expert taxonomist assigning documents to concepts."

        user_prompt = f"""Context: We have a document that doesn't clearly belong to any existing cluster. 
We need to assign it to the most appropriate concept.

Document:
{doc_text}

Candidate Concepts:
{candidates_text}

Task: Which concept does this document best fit into? Consider the core topic and methodology.

Output Format (JSON):
{{
  "best_match_id": 1 or 2 or 3 (or 0 if none fit),
  "reason": "Brief explanation"
}}"""

        try:
            result = self._retry_llm_and_parse_json(system_prompt, user_prompt)

            match_id = result.get('best_match_id', 0)

            if 1 <= match_id <= len(top_candidates):
                return top_candidates[match_id - 1][1]
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to assign outlier: {e}")
            return None

    def _print_statistics(self):
        """Print update statistics"""
        logger.info("=" * 60)
        logger.info("Update Statistics")
        logger.info("=" * 60)
        logger.info(f"Splits performed: {self.stats['splits']}")
        logger.info(f"Merges performed: {self.stats['merges']}")
        logger.info(f"Definition corrections: {self.stats['drift_corrections']}")
        logger.info(f"Nodes pruned: {self.stats['pruned']}")
        logger.info(f"Outliers reassigned: {self.stats['outlier_reassignments']}")
        logger.info(f"Total LLM calls: {self.stats['llm_calls']}")
        logger.info("=" * 60)


def main_cli():
    """Command-line interface for incremental updates"""
    import argparse
    import pickle
    from pathlib import Path
    from dataloader import load_graph_dataset
    from taxonomy_initialize import LLMClient, CoreSetSampler
    from taxonomy import TaxonomySaver

    parser = argparse.ArgumentParser(
        description='Incrementally update taxonomy with new embeddings'
    )

    # Input arguments
    parser.add_argument('--taxonomy_path', type=str, required=True,
                        help='Path to existing taxonomy pickle file')
    parser.add_argument('--old_embeddings_path', type=str, required=True,
                        help='Path to old embeddings numpy file')
    parser.add_argument('--new_embeddings_path', type=str, required=True,
                        help='Path to new embeddings numpy file')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name for loading new embeddings')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Path prefix for data')

    # LLM arguments
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for LLM service')
    parser.add_argument('--model', type=str, default='DeepSeek-V3',
                        help='LLM model name (default: DeepSeek-V3)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Base URL for LLM API (optional)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='LLM sampling temperature')
    parser.add_argument('--cache_dir', type=str, default='./llm_cache',
                        help='Directory for LLM cache')
    parser.add_argument('--llm_max_retries', type=int, default=3,
                        help='Maximum retries for LLM API calls (default: 3)')

    # Update parameters
    parser.add_argument('--cosine_threshold', type=float, default=0.5,
                        help='Minimum cosine similarity for assignment')
    parser.add_argument('--merge_threshold', type=float, default=0.9,
                        help='Cosine similarity threshold for merging')
    parser.add_argument('--variance_ratio', type=float, default=1.5,
                        help='Variance increase ratio for split detection')
    parser.add_argument('--size_ratio', type=float, default=2.5,
                        help='Size increase ratio for split detection')
    parser.add_argument('--turnover_threshold', type=float, default=0.5,
                        help='Document turnover threshold for drift detection')
    parser.add_argument('--min_cluster_size', type=int, default=5,
                        help='Minimum cluster size before pruning')

    # Sampling arguments
    parser.add_argument('--sampling_method', type=str, default='centroid',
                        choices=['centroid', 'degree', 'pagerank'],
                        help='Core-set sampling method')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of samples per cluster')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output_updated',
                        help='Output directory for updated taxonomy')
    parser.add_argument('--output_format', type=str, nargs='+',
                        default=['json', 'text', 'assignments', 'pickle'],
                        choices=['json', 'json_full', 'text', 'assignments', 'pickle'],
                        help='Output formats')

    args = parser.parse_args()

    # Load existing taxonomy
    logger.info(f"Loading existing taxonomy from {args.taxonomy_path}")
    with open(args.taxonomy_path, 'rb') as f:
        root = pickle.load(f)

    # Load old embeddings
    logger.info(f"Loading old embeddings from {args.old_embeddings_path}")
    old_embeddings = np.load(args.old_embeddings_path)

    # Load new embeddings
    logger.info(f"Loading new embeddings from {args.old_embeddings_path}")
    new_embeddings = np.load(args.new_embeddings_path)

    # Load graph data
    logger.info(f"Loading graph dataset: {args.dataset}")
    data = load_graph_dataset(
        dataset_name=args.dataset,
        device='cpu',
        path_prefix=args.data_path,
        re_split=0
    )

    old_embeddings = normalize(old_embeddings, norm='l2')
    new_embeddings = normalize(new_embeddings, norm='l2')

    edge_index = data.edge_index.numpy()
    raw_texts = data.raw_texts

    logger.info(f"Old embeddings shape: {old_embeddings.shape}")
    logger.info(f"New embeddings shape: {new_embeddings.shape}")

    # Initialize components
    llm_client = LLMClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        cache_dir=str(Path(args.cache_dir) / args.dataset)
    )

    sampler = CoreSetSampler(
        method=args.sampling_method,
        top_k=args.top_k
    )

    updater = TaxonomyUpdater(
        llm_client=llm_client,
        sampler=sampler,
        cosine_threshold=args.cosine_threshold,
        merge_threshold=args.merge_threshold,
        variance_increase_ratio=args.variance_ratio,
        size_increase_ratio=args.size_ratio,
        turnover_threshold=args.turnover_threshold,
        min_cluster_size=args.min_cluster_size,
        llm_max_retries=args.llm_max_retries
    )

    # Perform incremental update
    updated_root = updater.update_taxonomy(
        root=root,
        old_embeddings=old_embeddings,
        new_embeddings=new_embeddings,
        raw_texts=raw_texts,
        edge_index=edge_index
    )

    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export results
    logger.info("Exporting updated taxonomy...")

    if 'json' in args.output_format:
        TaxonomySaver.to_json(
            updated_root,
            output_path=str(output_dir / 'taxonomy_updated.json')
        )

    if 'json_full' in args.output_format:
        TaxonomySaver.to_json_with_docs(
            updated_root,
            output_path=str(output_dir / 'taxonomy_updated_full.json')
        )

    if 'text' in args.output_format:
        TaxonomySaver.to_text(
            updated_root,
            output_path=str(output_dir / 'taxonomy_updated.txt')
        )

    if 'assignments' in args.output_format:
        TaxonomySaver.to_assignments(
            updated_root,
            output_path=str(output_dir / 'assignments_updated.tsv')
        )

    if 'pickle' in args.output_format:
        TaxonomySaver.to_pickle(
            updated_root,
            output_path=str(output_dir / 'taxonomy_updated.pkl')
        )

    logger.info(f"Updated taxonomy saved to: {output_dir}")
    logger.info(json.dumps(TaxonomyQuery(root).get_statistics(node=root), indent=4, ensure_ascii=False))
    logger.info(json.dumps(TaxonomyQuery(updated_root).get_statistics(node=updated_root), indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main_cli()
