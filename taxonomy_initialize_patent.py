"""
LLM-driven Recursive Semantic Induction

Prompt for patent

This module implements a hierarchical clustering approach for document graphs
(e.g., citation networks) that generates explicit semantic labels using LLMs.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import json_repair
import numpy as np
import torch
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from dataloader import load_graph_dataset
from taxonomy import ConceptNode, TaxonomySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMCache:
    """Cache mechanism for LLM API calls to reduce costs"""

    def __init__(self, cache_dir: str = "./llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LLM cache at {self.cache_dir}")

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model"""
        content = f"{model}||{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Retrieve cached response"""
        cache_key = self._get_cache_key(prompt, model)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    logger.info(f"Cache hit for key {cache_key[:8]}...")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def set(self, prompt: str, model: str, response: str):
        """Store response in cache"""
        cache_key = self._get_cache_key(prompt, model)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(response, f)
                logger.info(f"Cached response for key {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


class LLMClient:
    """Client for interacting with LLM APIs"""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None,
                 temperature: float = 0.1, cache_dir: str = "./llm_cache"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.cache = LLMCache(cache_dir)

        # Configure OpenAI client
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        logger.info(f"Initialized LLM client with model: {model}")

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API with caching"""

        # Create full prompt for cache key
        full_prompt = f"SYSTEM: {system_prompt}\nUSER: {user_prompt}"

        # Check cache first
        cached_response = self.cache.get(full_prompt, self.model)
        if cached_response:
            return cached_response

        # Make API call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"}  # Enable JSON mode if supported
            )

            result = response.choices[0].message.content

            # Cache the response
            self.cache.set(full_prompt, self.model, result)

            return result

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise


class CoreSetSampler:
    """Sample representative documents from clusters"""

    def __init__(self, method: str = 'centroid', top_k: int = 5):
        """
        Initialize sampler

        Args:
            method: Sampling method ('centroid', 'degree', 'pagerank')
            top_k: Number of samples to select per cluster
        """
        self.method = method
        self.top_k = top_k
        logger.info(f"Initialized CoreSetSampler with method: {method}, top_k: {top_k}")

    def sample(self, embeddings: np.ndarray, doc_indices: List[int],
               cluster_center: np.ndarray, edge_index: Optional[np.ndarray] = None) -> List[int]:
        """
        Sample representative documents from a cluster

        Args:
            embeddings: Document embeddings (N x D)
            doc_indices: Indices of documents in this cluster
            cluster_center: Center of the cluster
            edge_index: Graph edge index for graph-based methods

        Returns:
            List of selected document indices
        """
        if len(doc_indices) <= self.top_k:
            return doc_indices

        cluster_embeddings = embeddings[doc_indices]

        if self.method == 'centroid':
            return self._sample_by_centroid(doc_indices, cluster_embeddings,
                                            cluster_center)
        elif self.method == 'degree':
            return self._sample_by_degree(doc_indices, cluster_embeddings,
                                          cluster_center, edge_index)
        elif self.method == 'pagerank':
            return self._sample_by_pagerank(doc_indices, cluster_embeddings,
                                            cluster_center, edge_index)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

    def _sample_by_centroid(self, doc_indices: List[int],
                            cluster_embeddings: np.ndarray,
                            cluster_center: np.ndarray) -> List[int]:
        """Sample documents closest to cluster center"""
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        top_indices = np.argsort(distances)[:self.top_k]
        return [doc_indices[i] for i in top_indices]

    def _sample_by_degree(self, doc_indices: List[int],
                          cluster_embeddings: np.ndarray,
                          cluster_center: np.ndarray,
                          edge_index: np.ndarray) -> List[int]:
        """Sample documents with high degree, constrained by centroid proximity"""
        # Get top 2*k candidates by centroid distance
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        top_2k_indices = np.argsort(distances)[:self.top_k * 2]
        top_2k_docs = [doc_indices[i] for i in top_2k_indices]

        # Calculate degrees for candidates
        degrees = self._calculate_degrees(top_2k_docs, edge_index)

        # Select top k by degree
        top_k_by_degree = np.argsort(degrees)[-self.top_k:]
        return [top_2k_docs[i] for i in top_k_by_degree]

    def _sample_by_pagerank(self, doc_indices: List[int],
                            cluster_embeddings: np.ndarray,
                            cluster_center: np.ndarray,
                            edge_index: np.ndarray) -> List[int]:
        """Sample documents with high PageRank in cluster subgraph"""
        # Get top 2*k candidates by centroid distance
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        top_2k_indices = np.argsort(distances)[:self.top_k * 2]
        top_2k_docs = [doc_indices[i] for i in top_2k_indices]

        # Calculate PageRank for candidates
        pagerank_scores = self._calculate_pagerank(top_2k_docs, edge_index)

        # Select top k by PageRank
        top_k_by_pr = np.argsort(pagerank_scores)[-self.top_k:]
        return [top_2k_docs[i] for i in top_k_by_pr]

    def _calculate_degrees(self, doc_indices: List[int],
                           edge_index: np.ndarray) -> np.ndarray:
        """Calculate node degrees for given documents"""
        doc_set = set(doc_indices)
        degrees = np.zeros(len(doc_indices))

        for i, doc in enumerate(doc_indices):
            # Count edges where both endpoints are in the cluster
            mask = (edge_index[0] == doc) | (edge_index[1] == doc)
            edges = edge_index[:, mask]
            degree = np.sum((np.isin(edges[0], list(doc_set))) &
                            (np.isin(edges[1], list(doc_set))))
            degrees[i] = degree

        return degrees

    def _calculate_pagerank(self, doc_indices: List[int],
                            edge_index: np.ndarray,
                            damping: float = 0.85,
                            max_iter: int = 100) -> np.ndarray:
        """Calculate PageRank for cluster subgraph"""
        n = len(doc_indices)
        doc_to_idx = {doc: i for i, doc in enumerate(doc_indices)}

        # Build adjacency matrix for subgraph
        adj_matrix = np.zeros((n, n))
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in doc_to_idx and dst in doc_to_idx:
                adj_matrix[doc_to_idx[src], doc_to_idx[dst]] = 1

        # Calculate out-degrees
        out_degrees = adj_matrix.sum(axis=1)
        out_degrees[out_degrees == 0] = 1  # Avoid division by zero

        # Normalize adjacency matrix
        transition_matrix = adj_matrix / out_degrees[:, np.newaxis]

        # Power iteration
        pagerank = np.ones(n) / n
        for _ in range(max_iter):
            pagerank_new = (1 - damping) / n + damping * transition_matrix.T @ pagerank
            if np.allclose(pagerank, pagerank_new, atol=1e-6):
                break
            pagerank = pagerank_new

        return pagerank


class TaxonomyBuilder:
    """Main class for building hierarchical taxonomy"""

    def __init__(self, llm_client: LLMClient, sampler: CoreSetSampler,
                 max_depth: int = 3, min_docs: int = 50,
                 over_cluster_factor: int = 20,
                 llm_max_retries: int = 3):
        """
        Initialize taxonomy builder

        Args:
            llm_client: LLM client for API calls
            sampler: Core-set sampler
            max_depth: Maximum tree depth
            min_docs: Minimum documents per node for splitting
            over_cluster_factor: Number of initial clusters (or sqrt factor)
            llm_max_retries: Maximum retries for LLM parsing failures
        """
        self.llm_client = llm_client
        self.sampler = sampler
        self.max_depth = max_depth
        self.min_docs = min_docs
        self.over_cluster_factor = over_cluster_factor
        self.llm_max_retries = llm_max_retries

        logger.info(f"Initialized TaxonomyBuilder: max_depth={max_depth}, "
                    f"min_docs={min_docs}, over_cluster_factor={over_cluster_factor}")

    def build_taxonomy(self, embeddings: np.ndarray, raw_texts: List[str],
                       edge_index: np.ndarray) -> ConceptNode:
        """
        Build complete taxonomy tree

        Args:
            embeddings: Document embeddings (N x D)
            raw_texts: Raw text content of documents
            edge_index: Graph edge index (2 x E)

        Returns:
            Root node of the taxonomy tree
        """
        logger.info("Starting taxonomy building...")

        # Create root node
        root = ConceptNode(
            name="Root",
            definition="All documents",
            doc_indices=list(range(len(raw_texts))),
            depth=0
        )

        # Recursively build tree
        self._recursive_split(root, embeddings, raw_texts, edge_index)

        logger.info("Taxonomy building completed!")
        return root

    def _recursive_split(self, parent: ConceptNode, embeddings: np.ndarray,
                         raw_texts: List[str], edge_index: np.ndarray):
        """
        Recursively split a parent node into children

        Args:
            parent: Parent concept node
            embeddings: Document embeddings
            raw_texts: Raw text content
            edge_index: Graph edge index
        """
        doc_indices = parent.doc_indices
        depth = parent.depth

        logger.info(f"Processing node '{parent.name}' at depth {depth} "
                    f"with {len(doc_indices)} documents")

        # Check stopping conditions
        if (depth >= self.max_depth or
                len(doc_indices) < self.min_docs or
                not parent.split_needed):
            logger.info(f"Stopping split for '{parent.name}': "
                        f"depth={depth}, docs={len(doc_indices)}, "
                        f"split_needed={parent.split_needed}")
            return

        # Phase 1: Over-clustering
        clusters, centers = self._over_cluster(embeddings, doc_indices)

        # Phase 2 & 3: Core-set sampling and LLM concept generation
        candidate_concepts = self._generate_concepts(
            clusters, centers, embeddings, raw_texts, edge_index, parent
        )

        # Phase 4: LLM-driven refinement
        refined_concepts = self._refine_concepts(
            candidate_concepts, parent, embeddings
        )

        # Create child nodes
        for concept_data in refined_concepts:
            child = ConceptNode(
                name=concept_data['name'],
                definition=concept_data['definition'],
                doc_indices=concept_data['doc_indices'],
                parent=parent,
                depth=depth + 1,
                split_needed=concept_data.get('split_needed', True),
                prototype=concept_data.get('prototype')
            )
            parent.children.append(child)

            logger.info(f"Created child node '{child.name}' with "
                        f"{len(child.doc_indices)} documents")

        # Phase 5: Recursion
        for child in parent.children:
            self._recursive_split(child, embeddings, raw_texts, edge_index)

    def _over_cluster(self, embeddings: np.ndarray,
                      doc_indices: List[int]) -> Tuple[List[List[int]], np.ndarray]:
        """
        Phase 1: Perform over-clustering using Spherical K-means

        Args:
            embeddings: Document embeddings
            doc_indices: Indices of documents to cluster

        Returns:
            Tuple of (cluster assignments, cluster centers)
        """
        cluster_embeddings = embeddings[doc_indices]

        # Normalize embeddings for spherical k-means
        normalized_embeddings = normalize(cluster_embeddings, norm='l2')

        # Determine number of clusters
        n_clusters = min(
            self.over_cluster_factor,
            max(2, int(np.sqrt(len(doc_indices) / 50)))
        )
        n_clusters = min(n_clusters, len(doc_indices))  # Can't exceed number of docs

        logger.info(f"Performing over-clustering with k={n_clusters}")

        # Perform spherical k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_embeddings)
        centers = kmeans.cluster_centers_

        # Organize documents by cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(doc_indices[i])

        # Remove empty clusters
        clusters = [c for c in clusters if len(c) > 0]

        logger.info(f"Created {len(clusters)} non-empty clusters")

        return clusters, centers

    def _generate_concepts(self, clusters: List[List[int]],
                           centers: np.ndarray,
                           embeddings: np.ndarray,
                           raw_texts: List[str],
                           edge_index: np.ndarray,
                           parent: ConceptNode) -> List[Dict[str, Any]]:
        """
        Phase 2 & 3: Sample core-set and generate concepts via LLM

        Args:
            clusters: List of document index lists per cluster
            centers: Cluster centers
            embeddings: Document embeddings
            raw_texts: Raw text content
            edge_index: Graph edge index
            parent: Parent concept node

        Returns:
            List of candidate concepts with metadata
        """
        candidate_concepts = []

        for cluster_id, cluster_docs in enumerate(clusters):
            if len(cluster_docs) == 0:
                continue

            logger.info(f"Processing cluster {cluster_id + 1}/{len(clusters)} "
                        f"with {len(cluster_docs)} documents")

            # Sample representative documents
            sampled_indices = self.sampler.sample(
                embeddings=embeddings,
                doc_indices=cluster_docs,
                cluster_center=centers[cluster_id],
                edge_index=edge_index
            )

            # Generate concept via LLM
            concept_data = self._llm_generate_concept(
                sampled_indices, raw_texts, parent
            )

            # Add cluster information
            concept_data['doc_indices'] = cluster_docs
            concept_data['prototype'] = embeddings[cluster_docs].mean(axis=0)
            concept_data['cluster_id'] = cluster_id

            candidate_concepts.append(concept_data)

        return candidate_concepts

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

    def _llm_generate_concept(self, sampled_indices: List[int],
                              raw_texts: List[str],
                              parent: ConceptNode) -> Dict[str, Any]:
        """
        Generate concept using LLM (Prompt 1)

        Args:
            sampled_indices: Indices of sampled documents
            raw_texts: Raw text content
            parent: Parent concept node

        Returns:
            Dictionary with concept_name, definition, and split_needed
        """
        # Prepare document texts
        doc_texts = []
        for i, idx in enumerate(sampled_indices, 1):
            text = raw_texts[idx][:500]  # Truncate for token efficiency
            doc_texts.append(f"{i}. {text}")

        docs_str = "\n".join(doc_texts)

        # Build prompts
        system_prompt = "You are an expert taxonomist assisting in organizing scientific literature."

        parent_info = f'"{parent.name}" ({parent.definition})' if parent.depth > 0 else '"All Documents" (Root category)'

        user_prompt = f"""Context: We are currently exploring sub-topics under the parent category: {parent_info}.

Task: Below are {len(sampled_indices)} representative patents from a specific cluster of documents within this parent category.
Your goal is to summarize the common research topic shared by these patents into a specific sub-concept.

Additionally, determine whether this sub-concept is semantically coherent and focused enough that it should NOT be further subdivided. 
If the patents clearly belong to a single, narrow topic with no distinct sub-directions, set "concept_split_needed" to false. 
If there appear to be multiple distinct sub-themes or methodological directions, set it to true.

Input patents:
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
                # 'split_needed': concept_data.get('concept_split_needed', True)
                'split_needed': True
            }
        except Exception as e:
            logger.error(f"Failed to generate concept after retries: {e}")
            return {
                'name': 'Cluster Concept',
                'definition': 'Auto-generated concept (parsing failed)',
                'split_needed': True
            }

    def _refine_concepts(self, candidate_concepts: List[Dict[str, Any]],
                         parent: ConceptNode,
                         embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """
        Phase 4: Refine concepts using LLM (Prompt 2)

        Args:
            candidate_concepts: List of candidate concepts
            parent: Parent concept node
            embeddings: Document embeddings

        Returns:
            List of refined concepts
        """
        if len(candidate_concepts) <= 1:
            return candidate_concepts

        logger.info(f"Refining {len(candidate_concepts)} candidate concepts")

        # Prepare concept list for LLM
        concept_list = []
        for i, concept in enumerate(candidate_concepts, 1):
            concept_list.append(
                f"{i} | {concept['name']} | {concept['definition']}"
            )

        concepts_str = "\n".join(concept_list)

        # Build prompts
        system_prompt = "You are a senior editor optimizing a hierarchical taxonomy."

        parent_info = f'"{parent.name}"' if parent.depth > 0 else '"All Documents"'

        user_prompt = f"""Context: Under the parent patent technology category {parent_info}, we identified {len(candidate_concepts)} candidate sub-concepts based on patent clustering. However, the clustering algorithm may have produced:
1. Duplicates/Synonyms (e.g., "Lithium-Ion Battery Cathodes" and "Li-Ion Battery Positive Electrodes" should be MERGED).
2. Overly fragmented topics (e.g., "OLED display driver circuits" and "AMOLED power control" might be MERGED into "OLED Display Control Systems").
3. Outliers/Irrelevant topics (e.g., an Agricultural Machinery patent topic appearing in a Semiconductor Manufacturing list should be PRUNED).

Candidate Sub-concepts:
ID | Name | Definition
{concepts_str}

Task: Review the list and output a set of operations to refine this level of the patent taxonomy.
Allowed Operations:
- MERGE: Combine IDs [id1, id2, ...] into a new, broader technical concept. Provide a new Name and Definition.
- PRUNE: Remove ID [id] if it is clearly off-topic, too vague, or low quality.
- KEEP: Keep ID [id] as is (or rename it slightly for clarity).

Output Format (JSON list):
[
  {{"op": "MERGE", "source_ids": [1, 5, 8], "new_name": "Lithium-Ion Battery Electrodes", "new_def": "...", "reason": "Overlapping patent technical features"}},
  {{"op": "PRUNE", "target_id": 3, "reason": "Unrelated agricultural technology patent"}},
  {{"op": "KEEP", "target_id": 2, "new_name": "Semiconductor Wafer Cleaning"}}
]"""

        try:
            operations = self._retry_llm_and_parse_json(system_prompt, user_prompt)
            if isinstance(operations, dict):
                tmp = iter(operations.values()).__next__()
                if isinstance(tmp, list):
                    # { "operations": [ {}, {} ] }
                    operations = tmp
                elif isinstance(tmp, str):
                    # {'op': 'PRUNE', ... }
                    operations = [operations]

            return self._execute_operations(operations, candidate_concepts, embeddings)
        except Exception as e:
            logger.error(f"Failed to parse refinement response after retries: {e}")
            logger.warning("Keeping all candidate concepts without refinement")
            return candidate_concepts

    def _execute_operations(self, operations: List[Dict[str, Any]],
                            candidate_concepts: List[Dict[str, Any]],
                            embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """
        Execute refinement operations (MERGE, PRUNE, KEEP)

        Args:
            operations: List of operations from LLM
            candidate_concepts: Original candidate concepts
            embeddings: Document embeddings

        Returns:
            List of refined concepts
        """
        # Track which concepts are processed
        processed = set()
        refined_concepts = []
        unassigned_docs = []

        for op in operations:
            op_type = op.get('op', '').upper()

            if op_type == 'MERGE':
                source_ids = op.get('source_ids', [])
                # Convert to 0-indexed
                source_indices = [i - 1 for i in source_ids if 0 < i <= len(candidate_concepts)]

                if len(source_indices) < 2:
                    logger.warning(f"Invalid MERGE operation: {op}")
                    continue

                # Merge documents
                merged_docs = []
                for idx in source_indices:
                    if idx not in processed:
                        merged_docs.extend(candidate_concepts[idx]['doc_indices'])
                        processed.add(idx)

                # Create merged concept
                merged_concept = {
                    'name': op.get('new_name', 'Merged Concept'),
                    'definition': op.get('new_def', 'Merged concept'),
                    'doc_indices': merged_docs,
                    'prototype': embeddings[merged_docs].mean(axis=0),
                    'split_needed': True  # Merged concepts may need further splitting
                }

                refined_concepts.append(merged_concept)
                logger.info(f"MERGED concepts {source_ids} into '{merged_concept['name']}'")

            elif op_type == 'PRUNE':
                target_id = op.get('target_id')
                if isinstance(target_id, list):
                    target_id = target_id[0] if target_id else None

                if target_id and 0 < target_id <= len(candidate_concepts):
                    target_idx = target_id - 1
                    if target_idx not in processed:
                        # Mark documents as unassigned for reassignment
                        unassigned_docs.extend(candidate_concepts[target_idx]['doc_indices'])
                        processed.add(target_idx)
                        logger.info(f"PRUNED concept {target_id}: {op.get('reason', 'No reason')}")

            elif op_type == 'KEEP':
                target_id = op.get('target_id')
                if isinstance(target_id, list):
                    target_id = target_id[0] if target_id else None

                if target_id and 0 < target_id <= len(candidate_concepts):
                    target_idx = target_id - 1
                    if target_idx not in processed:
                        concept = candidate_concepts[target_idx].copy()

                        # Optionally rename
                        new_name = op.get('new_name')
                        if new_name and new_name != "null":
                            concept['name'] = new_name

                        refined_concepts.append(concept)
                        processed.add(target_idx)
                        logger.info(f"KEPT concept {target_id}: '{concept['name']}'")

        # Keep any unprocessed concepts
        for i, concept in enumerate(candidate_concepts):
            if i not in processed:
                refined_concepts.append(concept)
                logger.info(f"Auto-kept unprocessed concept: '{concept['name']}'")

        # Reassign pruned documents to nearest cluster
        if unassigned_docs and refined_concepts:
            logger.info(f"Reassigning {len(unassigned_docs)} pruned documents")
            unassigned_embeddings = embeddings[unassigned_docs]

            for doc_idx, doc_emb in zip(unassigned_docs, unassigned_embeddings):
                # Find nearest prototype
                min_dist = float('inf')
                nearest_concept = None

                for concept in refined_concepts:
                    if concept.get('prototype') is not None:
                        dist = np.linalg.norm(doc_emb - concept['prototype'])
                        if dist < min_dist:
                            min_dist = dist
                            nearest_concept = concept

                if nearest_concept:
                    nearest_concept['doc_indices'].append(doc_idx)

        return refined_concepts


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLM-driven Recursive Semantic Induction'
    )

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset name (default: cora)')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Path prefix for data (default: current directory)')

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

    # Sampling arguments
    parser.add_argument('--sampling_method', type=str, default='pagerank',
                        choices=['centroid', 'degree', 'pagerank'],
                        help='Core-set sampling method (default: centroid)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of samples per cluster (default: 5)')

    # Taxonomy building arguments
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Maximum depth of taxonomy tree (default: 3)')
    parser.add_argument('--min_docs', type=int, default=50,
                        help='Minimum documents per node for splitting (default: 50)')
    parser.add_argument('--over_cluster_factor', type=int, default=20,
                        help='Initial number of clusters for over-clustering (default: 20)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results (default: ./output)')
    parser.add_argument('--output_format', type=str, nargs='+',
                        default=['json', 'text', 'assignments', 'pickle'],
                        choices=['json', 'json_full', 'text', 'assignments', 'pickle'],
                        help='Output formats (default: json text assignments, pickle)')

    # Misc arguments
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()


def main():
    """Main execution function"""

    # Parse arguments
    args = parse_arguments()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Set random seed
    np.random.seed(args.random_seed)

    logger.info("=" * 60)
    logger.info("LLM-driven Hierarchical Document Clustering")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    data = load_graph_dataset(
        dataset_name=args.dataset,
        device='cpu',
        path_prefix=args.data_path,
        re_split=0
    )

    emb_dir = Path(args.data_path) / "datasets" / "roberta"
    emb_path = emb_dir / f"{args.dataset}.pt"
    if emb_path.exists():
        logger.info(f"Loading pre-computed text embeddings from {emb_path}")
        embeddings = torch.load(emb_path, map_location='cpu').numpy()
    else:
        logger.info(f"Using default node embeddings with")
        embeddings = data.x.numpy()

    embeddings = normalize(embeddings, norm='l2')

    # Extract data
    edge_index = data.edge_index.numpy()
    # embeddings = data.x.numpy()
    raw_texts = data.raw_texts

    logger.info(f"Loaded {len(raw_texts)} documents with {embeddings.shape[1]}-dim embeddings")
    logger.info(f"Graph has {edge_index.shape[1]} edges")

    # Initialize components
    logger.info("Initializing components...")

    llm_client = LLMClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        cache_dir=Path(args.cache_dir) / args.dataset
    )

    sampler = CoreSetSampler(
        method=args.sampling_method,
        top_k=args.top_k
    )

    builder = TaxonomyBuilder(
        llm_client=llm_client,
        sampler=sampler,
        max_depth=args.max_depth,
        min_docs=args.min_docs,
        over_cluster_factor=args.over_cluster_factor,
        llm_max_retries = args.llm_max_retries,
    )

    # Build taxonomy
    logger.info("Building taxonomy tree...")
    root = builder.build_taxonomy(
        embeddings=embeddings,
        raw_texts=raw_texts,
        edge_index=edge_index
    )

    # Export results
    logger.info("Exporting results...")

    if 'json' in args.output_format:
        # Save as JSON (human-readable)
        TaxonomySaver.to_json(
            root,
            output_path=str(output_dir / 'taxonomy_plm.json')
        )

    if 'json_full' in args.output_format:
        # Save with document indices (for reconstruction)
        TaxonomySaver.to_pickle(
            root,
            output_path=(output_dir / 'taxonomy_full_plm.json')
        )

    if 'text' in args.output_format:
        TaxonomySaver.to_text(
            root,
            output_path=str(output_dir / 'taxonomy_plm.txt')
        )

    if 'assignments' in args.output_format:
        TaxonomySaver.to_assignments(
            root,
            output_path=str(output_dir / 'assignments_plm.tsv')
        )

    if 'pickle' in args.output_format:
        # Save as pickle (fastest for loading)
        TaxonomySaver.to_pickle(
            root,
            output_path=(output_dir / 'taxonomy_plm.pkl')
        )

    # Print summary statistics
    logger.info("=" * 60)
    logger.info("Taxonomy Statistics")
    logger.info("=" * 60)

    def count_nodes(node: ConceptNode, depth_counts: Dict[int, int]):
        """Count nodes at each depth"""
        depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
        for child in node.children:
            count_nodes(child, depth_counts)

    def count_leaf_nodes(node: ConceptNode) -> int:
        """Count leaf nodes"""
        if not node.children:
            return 1
        return sum(count_leaf_nodes(child) for child in node.children)

    depth_counts = {}
    count_nodes(root, depth_counts)

    total_nodes = sum(depth_counts.values())
    leaf_nodes = count_leaf_nodes(root)

    logger.info(f"Total nodes: {total_nodes}")
    logger.info(f"Leaf nodes: {leaf_nodes}")
    logger.info(f"Max depth: {max(depth_counts.keys())}")

    for depth in sorted(depth_counts.keys()):
        logger.info(f"  Depth {depth}: {depth_counts[depth]} nodes")

    logger.info("=" * 60)
    logger.info("Process completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
