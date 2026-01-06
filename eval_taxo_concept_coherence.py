"""
eval_taxo_concept_coherence.py

Evaluate learned taxonomy quality using semantic coherence metrics
"""

import argparse
import logging
import pickle
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
import numpy as np
from tqdm import tqdm
import re

from taxonomy import TaxonomyLoader, TaxonomyQuery, ConceptNode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocess text for coherence evaluation"""

    def __init__(self, min_word_length=3, remove_stopwords=True):
        self.min_word_length = min_word_length
        self.remove_stopwords = remove_stopwords

        # Common English stopwords
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
            't', 'just', 'don', 'now', 'also', 'using', 'used', 'use', 'based'
        }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', ' ', text)

        # Split into words
        words = text.split()

        # Filter words
        filtered_words = []
        for word in words:
            # Check length
            if len(word) < self.min_word_length:
                continue

            # Check stopwords
            if self.remove_stopwords and word in self.stopwords:
                continue

            filtered_words.append(word)

        return filtered_words


class CTFIDFExtractor:
    """Extract keywords using Class-based TF-IDF"""

    def __init__(self, top_k=10):
        self.top_k = top_k
        self.preprocessor = TextPreprocessor()

    def extract_concept_keywords(
        self,
        concept_docs: List[str],
        all_concepts_docs: List[List[str]]
    ) -> List[str]:
        """
        Extract top-k keywords for a concept using c-TF-IDF

        Args:
            concept_docs: Documents belonging to this concept
            all_concepts_docs: List of document lists for all concepts

        Returns:
            List of top-k keywords
        """
        # Tokenize all documents
        concept_tokens = []
        for doc in concept_docs:
            concept_tokens.extend(self.preprocessor.tokenize(doc))

        # Term frequency in this concept
        tf = Counter(concept_tokens)

        # Document frequency across all concepts
        df = defaultdict(int)
        for concept_doc_list in all_concepts_docs:
            concept_vocab = set()
            for doc in concept_doc_list:
                concept_vocab.update(self.preprocessor.tokenize(doc))
            for word in concept_vocab:
                df[word] += 1

        # Calculate c-TF-IDF
        n_concepts = len(all_concepts_docs)
        ctfidf_scores = {}

        for word, freq in tf.items():
            # TF component (normalized by concept size)
            tf_score = freq / len(concept_tokens) if len(concept_tokens) > 0 else 0

            # IDF component
            idf_score = np.log(1 + n_concepts / df[word]) if df[word] > 0 else 0

            ctfidf_scores[word] = tf_score * idf_score

        # Get top-k keywords
        sorted_words = sorted(
            ctfidf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_keywords = [word for word, score in sorted_words[:self.top_k]]

        return top_keywords


class CoherenceEvaluator:
    """Evaluate topic coherence using NPMI"""

    def __init__(self, window_size=20):
        """
        Args:
            window_size: Size of sliding window for co-occurrence counting
        """
        self.window_size = window_size
        self.preprocessor = TextPreprocessor()

        # Statistics for NPMI calculation
        self.word_doc_freq = defaultdict(int)  # P(w_i)
        self.word_pair_doc_freq = defaultdict(int)  # P(w_i, w_j)
        self.total_docs = 0

        # Statistics for window-based co-occurrence
        self.word_window_count = defaultdict(int)
        self.word_pair_window_count = defaultdict(int)
        self.total_windows = 0

    def build_statistics(self, documents: List[str]):
        """
        Build co-occurrence statistics from documents

        Args:
            documents: List of text documents
        """
        logger.info(f"Building co-occurrence statistics from {len(documents)} documents...")

        self.total_docs = len(documents)

        for doc in tqdm(documents, desc="Processing documents"):
            tokens = self.preprocessor.tokenize(doc)

            # Document-level co-occurrence
            unique_words = set(tokens)
            for word in unique_words:
                self.word_doc_freq[word] += 1

            # Pairwise co-occurrence in document
            for i, w1 in enumerate(unique_words):
                for w2 in list(unique_words)[i+1:]:
                    pair = tuple(sorted([w1, w2]))
                    self.word_pair_doc_freq[pair] += 1

            # Window-based co-occurrence
            for i in range(len(tokens)):
                window_end = min(i + self.window_size, len(tokens))
                window_words = set(tokens[i:window_end])

                self.total_windows += 1

                for word in window_words:
                    self.word_window_count[word] += 1

                for j, w1 in enumerate(window_words):
                    for w2 in list(window_words)[j+1:]:
                        pair = tuple(sorted([w1, w2]))
                        self.word_pair_window_count[pair] += 1

        logger.info(f"Statistics built:")
        logger.info(f"  Unique words: {len(self.word_doc_freq)}")
        logger.info(f"  Word pairs: {len(self.word_pair_doc_freq)}")
        logger.info(f"  Total windows: {self.total_windows}")

    def calculate_npmi(self, word1: str, word2: str, use_windows=False) -> float:
        """
        Calculate NPMI between two words

        Args:
            word1: First word
            word2: Second word
            use_windows: Use window-based co-occurrence instead of document-based

        Returns:
            NPMI score in [-1, 1]
        """
        if use_windows:
            # Window-based NPMI
            p_w1 = self.word_window_count[word1] / self.total_windows if self.total_windows > 0 else 0
            p_w2 = self.word_window_count[word2] / self.total_windows if self.total_windows > 0 else 0

            pair = tuple(sorted([word1, word2]))
            p_w1_w2 = self.word_pair_window_count[pair] / self.total_windows if self.total_windows > 0 else 0
        else:
            # Document-based NPMI
            p_w1 = self.word_doc_freq[word1] / self.total_docs if self.total_docs > 0 else 0
            p_w2 = self.word_doc_freq[word2] / self.total_docs if self.total_docs > 0 else 0

            pair = tuple(sorted([word1, word2]))
            p_w1_w2 = self.word_pair_doc_freq[pair] / self.total_docs if self.total_docs > 0 else 0

        # Avoid division by zero
        if p_w1_w2 == 0 or p_w1 == 0 or p_w2 == 0:
            return 0.0

        # Calculate PMI
        pmi = np.log(p_w1_w2 / (p_w1 * p_w2))

        # Normalize by joint probability
        npmi = pmi / (-np.log(p_w1_w2))

        # Clip to valid range
        npmi = np.clip(npmi, -1.0, 1.0)

        return npmi

    def calculate_concept_coherence(
        self,
        keywords: List[str],
        use_windows=False
    ) -> float:
        """
        Calculate coherence score for a concept based on its keywords

        Args:
            keywords: List of keywords representing the concept
            use_windows: Use window-based NPMI

        Returns:
            Average NPMI score
        """
        if len(keywords) < 2:
            return 0.0

        npmi_scores = []

        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                npmi = self.calculate_npmi(keywords[i], keywords[j], use_windows)
                npmi_scores.append(npmi)

        return np.mean(npmi_scores) if npmi_scores else 0.0


class TaxonomyCoherenceEvaluator:
    """Evaluate taxonomy coherence"""

    def __init__(
        self,
        taxonomy_root: ConceptNode,
        documents: List[str],
        top_k_keywords=10,
        window_size=20,
        use_windows=False
    ):
        """
        Args:
            taxonomy_root: Root of taxonomy tree
            documents: All documents in the dataset
            top_k_keywords: Number of keywords to extract per concept
            window_size: Window size for co-occurrence
            use_windows: Use window-based NPMI instead of document-based
        """
        self.taxonomy_root = taxonomy_root
        self.documents = documents
        self.top_k_keywords = top_k_keywords
        self.use_windows = use_windows

        self.query = TaxonomyQuery(taxonomy_root)
        self.keyword_extractor = CTFIDFExtractor(top_k=top_k_keywords)
        self.coherence_evaluator = CoherenceEvaluator(window_size=window_size)

        # Build statistics
        self.coherence_evaluator.build_statistics(documents)

        # Extract keywords for all concepts
        self.concept_keywords = {}
        self._extract_all_keywords()

    def _get_concept_documents(self, node: ConceptNode) -> List[str]:
        """Get all documents belonging to a concept"""
        return [self.documents[idx] for idx in node.doc_indices]

    def _extract_all_keywords(self):
        """Extract keywords for all concepts in taxonomy"""
        logger.info("Extracting keywords for all concepts...")

        # Get all leaf nodes (they contain actual document assignments)
        leaf_nodes = self.query.get_leaf_nodes()

        # Collect documents for all leaf concepts
        all_concepts_docs = []
        for node in leaf_nodes:
            concept_docs = self._get_concept_documents(node)
            if concept_docs:  # Only include non-empty concepts
                all_concepts_docs.append(concept_docs)

        # Extract keywords for each concept
        for node in tqdm(leaf_nodes, desc="Extracting keywords"):
            concept_docs = self._get_concept_documents(node)

            if not concept_docs:
                self.concept_keywords[node.name] = []
                continue

            keywords = self.keyword_extractor.extract_concept_keywords(
                concept_docs,
                all_concepts_docs
            )

            self.concept_keywords[node.name] = keywords

            logger.debug(f"Concept '{node.name}': {keywords}")

    def evaluate(self) -> Dict:
        """
        Evaluate taxonomy coherence

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("=" * 60)
        logger.info("Evaluating Taxonomy Coherence")
        logger.info("=" * 60)

        leaf_nodes = self.query.get_leaf_nodes()

        # Calculate coherence for each concept
        concept_coherences = {}
        valid_concepts = 0

        for node in tqdm(leaf_nodes, desc="Calculating coherence"):
            keywords = self.concept_keywords.get(node.name, [])

            if len(keywords) < 2:
                continue

            coherence = self.coherence_evaluator.calculate_concept_coherence(
                keywords,
                use_windows=self.use_windows
            )

            concept_coherences[node.name] = {
                'coherence': coherence,
                'keywords': keywords,
                'num_docs': len(node.doc_indices),
                'depth': node.depth
            }

            valid_concepts += 1

        # Calculate overall statistics
        all_coherences = [c['coherence'] for c in concept_coherences.values()]

        results = {
            'overall_coherence_mean': np.mean(all_coherences) if all_coherences else 0.0,
            'overall_coherence_std': np.std(all_coherences) if all_coherences else 0.0,
            'overall_coherence_median': np.median(all_coherences) if all_coherences else 0.0,
            'num_concepts': len(leaf_nodes),
            'num_valid_concepts': valid_concepts,
            'concept_details': concept_coherences
        }

        # Calculate coherence by depth
        coherence_by_depth = defaultdict(list)
        for concept_info in concept_coherences.values():
            coherence_by_depth[concept_info['depth']].append(concept_info['coherence'])

        depth_statistics = {}
        for depth, coherences in coherence_by_depth.items():
            depth_statistics[depth] = {
                'mean': np.mean(coherences),
                                'std': np.std(coherences),
                'median': np.median(coherences),
                'count': len(coherences)
            }

        results['coherence_by_depth'] = depth_statistics

        # Log results
        logger.info("\n" + "=" * 60)
        logger.info("Coherence Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"Total concepts: {results['num_concepts']}")
        logger.info(f"Valid concepts (≥2 keywords): {results['num_valid_concepts']}")
        logger.info(f"\nOverall Coherence (NPMI):")
        logger.info(f"  Mean:   {results['overall_coherence_mean']:.4f}")
        logger.info(f"  Std:    {results['overall_coherence_std']:.4f}")
        logger.info(f"  Median: {results['overall_coherence_median']:.4f}")

        logger.info(f"\nCoherence by Depth:")
        for depth in sorted(depth_statistics.keys()):
            stats = depth_statistics[depth]
            logger.info(f"  Depth {depth}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                       f"(n={stats['count']})")

        # Top and bottom concepts by coherence
        sorted_concepts = sorted(
            concept_coherences.items(),
            key=lambda x: x[1]['coherence'],
            reverse=True
        )

        logger.info(f"\nTop-5 Most Coherent Concepts:")
        for i, (name, info) in enumerate(sorted_concepts[:5], 1):
            logger.info(f"  {i}. {name} (coherence={info['coherence']:.4f})")
            logger.info(f"     Keywords: {', '.join(info['keywords'])}")
            logger.info(f"     Docs: {info['num_docs']}, Depth: {info['depth']}")

        logger.info(f"\nBottom-5 Least Coherent Concepts:")
        for i, (name, info) in enumerate(sorted_concepts[-5:][::-1], 1):
            logger.info(f"  {i}. {name} (coherence={info['coherence']:.4f})")
            logger.info(f"     Keywords: {', '.join(info['keywords'])}")
            logger.info(f"     Docs: {info['num_docs']}, Depth: {info['depth']}")

        return results

    def save_results(self, output_path: str):
        """Save evaluation results to JSON"""
        results = self.evaluate()

        # Convert to JSON-serializable format
        output = {
            'overall_coherence_mean': float(results['overall_coherence_mean']),
            'overall_coherence_std': float(results['overall_coherence_std']),
            'overall_coherence_median': float(results['overall_coherence_median']),
            'num_concepts': results['num_concepts'],
            'num_valid_concepts': results['num_valid_concepts'],
            'coherence_by_depth': {
                str(depth): {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'median': float(stats['median']),
                    'count': int(stats['count'])
                }
                for depth, stats in results['coherence_by_depth'].items()
            },
            'concept_details': {
                name: {
                    'coherence': float(info['coherence']),
                    'keywords': info['keywords'],
                    'num_docs': int(info['num_docs']),
                    'depth': int(info['depth'])
                }
                for name, info in results['concept_details'].items()
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to {output_path}")


def load_documents_from_dataset(dataset_name: str, data_path: str) -> List[str]:
    """Load raw text documents from dataset"""
    from dataloader import load_graph_dataset

    logger.info(f"Loading documents from dataset: {dataset_name}")

    data = load_graph_dataset(
        dataset_name=dataset_name,
        device='cpu',
        path_prefix=data_path,
        re_split=0
    )

    if hasattr(data, 'raw_texts'):
        documents = data.raw_texts
    else:
        raise ValueError(f"Dataset {dataset_name} does not have raw_texts attribute")

    logger.info(f"Loaded {len(documents)} documents")

    return documents


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate taxonomy coherence using NPMI'
    )

    # Required arguments
    parser.add_argument('--taxonomy_path', type=str, required=True,
                       help='Path to taxonomy pickle file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., cora, citeseer)')

    # Optional arguments
    parser.add_argument('--data_path', type=str, default='.',
                       help='Path prefix for data')
    parser.add_argument('--output_dir', type=str, default='./taxonomy_eval_results/coherence',
                       help='Output directory for results')
    parser.add_argument('--top_k_keywords', type=int, default=10,
                       help='Number of keywords to extract per concept')
    parser.add_argument('--window_size', type=int, default=20,
                       help='Window size for co-occurrence counting')
    parser.add_argument('--use_windows', type=int, default=0,
                       help='Use window-based NPMI (1) or document-based (0)')

    args = parser.parse_args()

    # Setup output directory
    taxonomy_version = Path(args.taxonomy_path).stem
    output_dir = Path(args.output_dir) / args.dataset / taxonomy_version
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Taxonomy Coherence Evaluation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Taxonomy: {args.taxonomy_path}")
    logger.info(f"Top-K Keywords: {args.top_k_keywords}")
    logger.info(f"Window Size: {args.window_size}")
    logger.info(f"Use Windows: {bool(args.use_windows)}")

    # Load taxonomy
    logger.info("\nLoading taxonomy...")
    taxonomy_root = TaxonomyLoader.from_pickle(args.taxonomy_path)

    # Load documents
    logger.info("\nLoading documents...")
    documents = load_documents_from_dataset(args.dataset, args.data_path)

    # Initialize evaluator
    logger.info("\nInitializing evaluator...")
    evaluator = TaxonomyCoherenceEvaluator(
        taxonomy_root=taxonomy_root,
        documents=documents,
        top_k_keywords=args.top_k_keywords,
        window_size=args.window_size,
        use_windows=bool(args.use_windows)
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Save results
    output_path = output_dir / 'coherence_results.json'
    evaluator.save_results(str(output_path))

    # Save keywords for inspection
    keywords_path = output_dir / 'concept_keywords.txt'
    with open(keywords_path, 'w', encoding='utf-8') as f:
        f.write("Concept Keywords\n")
        f.write("=" * 80 + "\n\n")

        for concept_name in sorted(evaluator.concept_keywords.keys()):
            keywords = evaluator.concept_keywords[concept_name]
            info = results['concept_details'].get(concept_name, {})
            coherence = info.get('coherence', 0.0)
            num_docs = info.get('num_docs', 0)
            depth = info.get('depth', 0)

            f.write(f"Concept: {concept_name}\n")
            f.write(f"  Coherence: {coherence:.4f}\n")
            f.write(f"  Documents: {num_docs}\n")
            f.write(f"  Depth: {depth}\n")
            f.write(f"  Keywords: {', '.join(keywords)}\n")
            f.write("\n")

    logger.info(f"Concept keywords saved to {keywords_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

#!/bin/bash

# DATASET="cora"
# TAXONOMY_PATH="./output/cora/taxonomy_final.pkl"
# DATA_PATH="."
# OUTPUT_DIR="./taxonomy_eval_results/coherence"
#
# python evaluate_taxonomy_topic_coherence.py \
#     --taxonomy_path ${TAXONOMY_PATH} \
#     --dataset ${DATASET} \
#     --data_path ${DATA_PATH} \
#     --output_dir ${OUTPUT_DIR} \
#     --top_k_keywords 10 \
#     --window_size 20 \
#     --use_windows 0
