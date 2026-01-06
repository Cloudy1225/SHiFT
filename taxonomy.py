"""
taxonomy.py

Module for loading and manipulating saved taxonomy trees
"""

import json
import logging
import pickle
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConceptNode:
    """Represents a concept node in the taxonomy tree"""
    name: str
    definition: str
    doc_indices: List[int]
    parent: Optional['ConceptNode'] = None
    children: List['ConceptNode'] = None
    depth: int = 0
    split_needed: bool = True
    prototype: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class TaxonomyLoader:
    """Load and reconstruct taxonomy trees from saved files"""

    @staticmethod
    def from_json(json_path: str) -> ConceptNode:
        """
        Load taxonomy from JSON file

        Args:
            json_path: Path to JSON file

        Returns:
            Root ConceptNode of the taxonomy
        """
        logger.info(f"Loading taxonomy from JSON: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        root = TaxonomyLoader._dict_to_node(data)

        logger.info(f"Loaded taxonomy with {TaxonomyLoader._count_nodes(root)} nodes")
        return root

    @staticmethod
    def from_pickle(pickle_path: str) -> ConceptNode:
        """
        Load taxonomy from pickle file

        Args:
            pickle_path: Path to pickle file

        Returns:
            Root ConceptNode of the taxonomy
        """
        logger.info(f"Loading taxonomy from pickle: {pickle_path}")

        with open(pickle_path, 'rb') as f:
            root = pickle.load(f)

        logger.info(f"Loaded taxonomy with {TaxonomyLoader._count_nodes(root)} nodes")
        return root

    @staticmethod
    def _dict_to_node(data: Dict[str, Any], parent: Optional[ConceptNode] = None) -> ConceptNode:
        """
        Recursively convert dictionary to ConceptNode

        Args:
            data: Dictionary representation of node
            parent: Parent node (None for root)

        Returns:
            Reconstructed ConceptNode
        """
        # Create node
        node = ConceptNode(
            name=data['name'],
            definition=data['definition'],
            doc_indices=[],  # Will be populated from children or assignments file
            parent=parent,
            depth=data.get('depth', 0),
            split_needed=data.get('split_needed', True)
        )

        # Recursively create children
        if 'children' in data:
            for child_data in data['children']:
                child = TaxonomyLoader._dict_to_node(child_data, parent=node)
                node.children.append(child)

        return node

    @staticmethod
    def load_with_assignments(json_path: str, assignments_path: str) -> ConceptNode:
        """
        Load taxonomy from JSON and populate doc_indices from assignments file
        Parent nodes will automatically aggregate documents from all children

        Args:
            json_path: Path to taxonomy JSON file
            assignments_path: Path to assignments TSV file

        Returns:
            Root ConceptNode with populated doc_indices (aggregated for parent nodes)
        """
        logger.info(f"Loading taxonomy with assignments")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  Assignments: {assignments_path}")

        # Load taxonomy structure
        root = TaxonomyLoader.from_json(json_path)

        # Load assignments
        assignments = TaxonomyLoader._load_assignments(assignments_path)

        # Create path-to-node mapping
        path_to_node = {}
        TaxonomyLoader._build_path_mapping(root, [], path_to_node)

        # Populate doc_indices for leaf nodes
        for doc_id, path in assignments:
            if path in path_to_node:
                path_to_node[path].doc_indices.append(doc_id)
            else:
                logger.warning(f"Path not found in taxonomy: {path}")

        # Aggregate documents from children to parents
        TaxonomyLoader._aggregate_documents(root)

        logger.info(f"Loaded taxonomy with assignments for {len(assignments)} documents")
        return root

    @staticmethod
    def _aggregate_documents(node: ConceptNode) -> List[int]:
        """
        Recursively aggregate document indices from children to parents

        Args:
            node: Current node

        Returns:
            List of all document indices in this subtree
        """
        if not node.children:
            # Leaf node - return its documents as-is
            return node.doc_indices

        # Non-leaf node - aggregate from all children
        all_docs = set()
        for child in node.children:
            child_docs = TaxonomyLoader._aggregate_documents(child)
            all_docs.update(child_docs)

        # Update node's doc_indices with aggregated documents
        node.doc_indices = sorted(list(all_docs))

        return node.doc_indices

    @staticmethod
    def _load_assignments(assignments_path: str) -> List[tuple]:
        """Load document assignments from TSV file"""
        assignments = []

        with open(assignments_path, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    doc_id = int(parts[0])
                    path = parts[1]
                    assignments.append((doc_id, path))

        return assignments

    @staticmethod
    def _build_path_mapping(node: ConceptNode, path: List[str],
                            mapping: Dict[str, ConceptNode]):
        """Build mapping from path string to node"""
        current_path = path + [node.name]
        path_str = " > ".join(current_path)
        mapping[path_str] = node

        for child in node.children:
            TaxonomyLoader._build_path_mapping(child, current_path, mapping)

    @staticmethod
    def _count_nodes(node: ConceptNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += TaxonomyLoader._count_nodes(child)
        return count


class TaxonomyQuery:
    """Query and navigate taxonomy trees"""

    def __init__(self, root: ConceptNode):
        """
        Initialize taxonomy query interface

        Args:
            root: Root node of taxonomy
        """
        self.root = root
        self._node_cache = {}
        self._build_cache()

    def _build_cache(self):
        """Build cache for fast lookups"""

        def traverse(node: ConceptNode, path: List[str]):
            current_path = path + [node.name]
            path_str = " > ".join(current_path)

            self._node_cache[node.name.lower()] = node
            self._node_cache[path_str] = node

            for child in node.children:
                traverse(child, current_path)

        traverse(self.root, [])
        logger.info(f"Built cache with {len(self._node_cache)} entries")

    def find_concept(self, concept_name: str) -> Optional[ConceptNode]:
        """
        Find a concept node by name

        Args:
            concept_name: Name of concept to find

        Returns:
            ConceptNode if found, None otherwise
        """
        return self._node_cache.get(concept_name.lower())

    def find_by_path(self, path: str) -> Optional[ConceptNode]:
        """
        Find a concept node by full path

        Args:
            path: Full path (e.g., "Root > Machine Learning > Neural Networks")

        Returns:
            ConceptNode if found, None otherwise
        """
        return self._node_cache.get(path)

    def get_path(self, node: ConceptNode) -> str:
        """
        Get full path to a node

        Args:
            node: Target node

        Returns:
            Full path string
        """
        path = []
        current = node

        while current is not None:
            path.insert(0, current.name)
            current = current.parent

        return " > ".join(path)

    def get_siblings(self, node: ConceptNode) -> List[ConceptNode]:
        """
        Get sibling nodes

        Args:
            node: Target node

        Returns:
            List of sibling nodes (excluding the node itself)
        """
        if node.parent is None:
            return []

        return [child for child in node.parent.children if child != node]

    def get_ancestors(self, node: ConceptNode) -> List[ConceptNode]:
        """
        Get all ancestor nodes

        Args:
            node: Target node

        Returns:
            List of ancestor nodes (from root to parent)
        """
        ancestors = []
        current = node.parent

        while current is not None:
            ancestors.insert(0, current)
            current = current.parent

        return ancestors

    def get_descendants(self, node: ConceptNode) -> List[ConceptNode]:
        """
        Get all descendant nodes

        Args:
            node: Target node

        Returns:
            List of all descendant nodes
        """
        descendants = []

        def traverse(n: ConceptNode):
            for child in n.children:
                descendants.append(child)
                traverse(child)

        traverse(node)
        return descendants

    def get_leaf_nodes(self, node: Optional[ConceptNode] = None) -> List[ConceptNode]:
        """
        Get all leaf nodes under a node

        Args:
            node: Starting node (uses root if None)

        Returns:
            List of leaf nodes
        """
        if node is None:
            node = self.root

        leaves = []

        def traverse(n: ConceptNode):
            if not n.children:
                leaves.append(n)
            else:
                for child in n.children:
                    traverse(child)

        traverse(node)
        return leaves

    def get_documents_in_subtree(self, node: ConceptNode) -> List[int]:
        """
        Get all document IDs in a subtree

        Args:
            node: Root of subtree

        Returns:
            List of document IDs
        """
        doc_ids = set()

        def traverse(n: ConceptNode):
            doc_ids.update(n.doc_indices)
            for child in n.children:
                traverse(child)

        traverse(node)
        return sorted(list(doc_ids))

    def get_statistics(self, node: Optional[ConceptNode] = None) -> Dict[str, Any]:
        """
        Get statistics for a subtree

        Args:
            node: Root of subtree (uses root if None)

        Returns:
            Dictionary with statistics
        """
        if node is None:
            node = self.root

        descendants = self.get_descendants(node)
        leaves = self.get_leaf_nodes(node)
        all_docs = self.get_documents_in_subtree(node)

        # Calculate depth distribution
        depth_counts = {}
        for desc in descendants + [node]:
            depth_counts[desc.depth] = depth_counts.get(desc.depth, 0) + 1

        return {
            'root_name': node.name,
            'total_nodes': len(descendants) + 1,
            'leaf_nodes': len(leaves),
            'total_documents': len(all_docs),
            'max_depth': max(depth_counts.keys()) if depth_counts else node.depth,
            'depth_distribution': depth_counts
        }

    def search_by_keyword(self, keyword: str) -> List[ConceptNode]:
        """
        Search for concepts containing a keyword

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching nodes
        """
        keyword_lower = keyword.lower()
        matches = []

        def traverse(node: ConceptNode):
            if (keyword_lower in node.name.lower() or
                    keyword_lower in node.definition.lower()):
                matches.append(node)

            for child in node.children:
                traverse(child)

        traverse(self.root)
        return matches


class TaxonomyVisualizer:
    """Visualize taxonomy trees"""

    @staticmethod
    def print_tree(root: ConceptNode, max_depth: Optional[int] = None,
                   show_docs: bool = True):
        """
        Print taxonomy tree in ASCII format

        Args:
            root: Root node
            max_depth: Maximum depth to display (None for all)
            show_docs: Whether to show document counts
        """

        def print_node(node: ConceptNode, prefix: str = "", is_last: bool = True):
            if max_depth is not None and node.depth > max_depth:
                return

            # Prepare node display
            connector = "└── " if is_last else "├── "
            doc_info = f" ({len(node.doc_indices)} docs)" if show_docs else ""

            print(f"{prefix}{connector}{node.name}{doc_info}")
            print(f"{prefix}{'    ' if is_last else '│   '}    {node.definition}")

            # Print children
            if node.children:
                extension = "    " if is_last else "│   "
                for i, child in enumerate(node.children):
                    is_last_child = (i == len(node.children) - 1)
                    print_node(child, prefix + extension, is_last_child)

        print(f"\n{root.name} ({len(root.doc_indices)} docs)")
        print(f"  {root.definition}")

        for i, child in enumerate(root.children):
            is_last = (i == len(root.children) - 1)
            print_node(child, "", is_last)

        print()

    @staticmethod
    def to_graphviz(root: ConceptNode, output_path: str, format: str = 'pdf'):
        """
        Export taxonomy to Graphviz DOT format and render

        Args:
            root: Root node
            output_path: Output file path (without extension)
            format: Output format ('pdf', 'png', 'svg', etc.)
        """
        try:
            import graphviz
        except ImportError:
            logger.error("graphviz package not installed. Install with: pip install graphviz")
            return

        dot = graphviz.Digraph(comment='Taxonomy Tree')
        dot.attr(rankdir='TB')

        node_counter = [0]

        def add_node(node: ConceptNode, parent_id: Optional[str] = None):
            node_id = f"node_{node_counter[0]}"
            node_counter[0] += 1

            label = f"{node.name}\n({len(node.doc_indices)} docs)"
            dot.node(node_id, label, shape='box', style='rounded,filled',
                     fillcolor='lightblue' if node.children else 'lightgreen')

            if parent_id:
                dot.edge(parent_id, node_id)

            for child in node.children:
                add_node(child, node_id)

        add_node(root)

        # Render
        dot.render(output_path, format=format, cleanup=True)
        logger.info(f"Taxonomy visualization saved to {output_path}.{format}")


class TaxonomySaver:
    """Export taxonomy to various formats"""

    @staticmethod
    def to_pickle(root: ConceptNode, output_path: str):
        """
        Save taxonomy to pickle file (preserves all attributes)

        Args:
            root: Root node of taxonomy
            output_path: Path to save pickle file
        """
        logger.info(f"Saving taxonomy to pickle: {output_path}")

        with open(output_path, 'wb') as f:
            pickle.dump(root, f)

        logger.info("Taxonomy saved successfully")

    @staticmethod
    def to_json(root: ConceptNode, output_path: str):
        """Export taxonomy to JSON format"""

        def node_to_dict(node: ConceptNode) -> Dict[str, Any]:
            """Convert node to dictionary recursively"""
            return {
                'name': node.name,
                'definition': node.definition,
                'depth': node.depth,
                'num_documents': len(node.doc_indices),
                'split_needed': node.split_needed,
                'children': [node_to_dict(child) for child in node.children]
            }

        taxonomy_dict = node_to_dict(root)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(taxonomy_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported taxonomy to {output_path}")

    @staticmethod
    def to_json_with_docs(root: ConceptNode, output_path: str):
        """
        Save taxonomy to JSON including document indices

        Args:
            root: Root node of taxonomy
            output_path: Path to save JSON file
        """
        logger.info(f"Saving taxonomy with document indices to: {output_path}")

        def node_to_dict(node: ConceptNode) -> Dict[str, Any]:
            return {
                'name': node.name,
                'definition': node.definition,
                'depth': node.depth,
                'num_documents': len(node.doc_indices),
                'doc_indices': node.doc_indices,  # Include doc indices
                'split_needed': node.split_needed,
                'children': [node_to_dict(child) for child in node.children]
            }

        taxonomy_dict = node_to_dict(root)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(taxonomy_dict, f, indent=2, ensure_ascii=False)

        logger.info("Taxonomy saved successfully")

    @staticmethod
    def to_text(root: ConceptNode, output_path: str):
        """Export taxonomy to readable text format"""

        def node_to_text(node: ConceptNode, indent: int = 0) -> str:
            """Convert node to text recursively"""
            prefix = "  " * indent
            text = f"{prefix}- {node.name} ({len(node.doc_indices)} docs)\n"
            text += f"{prefix}  Definition: {node.definition}\n"

            for child in node.children:
                text += node_to_text(child, indent + 1)

            return text

        taxonomy_text = node_to_text(root)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Document Taxonomy\n")
            f.write("=" * 50 + "\n\n")
            f.write(taxonomy_text)

        logger.info(f"Exported taxonomy to {output_path}")

    @staticmethod
    def to_assignments(root: ConceptNode, output_path: str):
        """Export document-to-concept assignments"""

        def collect_assignments(node: ConceptNode, path: List[str]) -> List[Tuple[int, str]]:
            """Collect document assignments recursively"""
            current_path = path + [node.name]

            # If leaf node, assign documents
            if not node.children:
                path_str = " > ".join(current_path)
                return [(doc_idx, path_str) for doc_idx in node.doc_indices]

            # Otherwise recurse
            assignments = []
            for child in node.children:
                assignments.extend(collect_assignments(child, current_path))

            return assignments

        assignments = collect_assignments(root, [])

        # Sort by document index
        assignments.sort(key=lambda x: x[0])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DocumentID\tTaxonomyPath\n")
            for doc_idx, path in assignments:
                f.write(f"{doc_idx}\t{path}\n")

        logger.info(f"Exported assignments to {output_path}")


# Example usage functions
def example_load_and_query():
    """Example: Load taxonomy and perform queries"""

    # Method 1: Load from JSON only (no doc_indices)
    root = TaxonomyLoader.from_json('output/taxonomy.json')

    # Method 2: Load from JSON with assignments
    root = TaxonomyLoader.load_with_assignments(
        'output/taxonomy.json',
        'output/assignments.tsv'
    )

    # Method 3: Load from pickle (fastest, preserves everything)
    # root = TaxonomyLoader.from_pickle('output/taxonomy.pkl')

    # Create query interface
    query = TaxonomyQuery(root)

    # Find a concept
    ml_node = query.find_concept("Machine Learning")
    if ml_node:
        print(f"Found: {ml_node.name}")
        print(f"Definition: {ml_node.definition}")
        print(f"Documents: {len(ml_node.doc_indices)}")

    # Get statistics
    stats = query.get_statistics()
    print(f"\nTaxonomy Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Leaf nodes: {stats['leaf_nodes']}")
    print(f"  Total documents: {stats['total_documents']}")

    # Search by keyword
    matches = query.search_by_keyword("neural")
    print(f"\nConcepts containing 'neural': {len(matches)}")
    for node in matches:
        print(f"  - {query.get_path(node)}")

    # Get all documents in a subtree
    if ml_node:
        docs = query.get_documents_in_subtree(ml_node)
        print(f"\nDocuments under '{ml_node.name}': {len(docs)}")

    # Visualize
    TaxonomyVisualizer.print_tree(root, max_depth=2)

def example_modify_and_save():
    """Example: Load, modify, and save taxonomy"""

    # Load taxonomy
    root = TaxonomyLoader.load_with_assignments(
        'output/taxonomy.json',
        'output/assignments.tsv'
    )

    query = TaxonomyQuery(root)

    # Find a node to modify
    node = query.find_concept("Neural Networks")
    if node:
        # Modify node
        node.definition = "Updated definition for neural networks"

        # Add a new child manually
        new_child = ConceptNode(
            name="Transformers",
            definition="Attention-based neural network architectures",
            doc_indices=[],  # Add specific doc IDs
            parent=node,
            depth=node.depth + 1
        )
        node.children.append(new_child)

    # Save modified taxonomy
    TaxonomySaver.to_pickle(root, 'output/taxonomy_modified.pkl')
    TaxonomySaver.to_json_with_docs(root, 'output/taxonomy_modified.json')

    print("Modified taxonomy saved!")

def example_compare_taxonomies():
    """Example: Compare two taxonomies"""

    taxonomy1 = TaxonomyLoader.from_json('output/taxonomy_v1.json')
    taxonomy2 = TaxonomyLoader.from_json('output/taxonomy_v2.json')

    query1 = TaxonomyQuery(taxonomy1)
    query2 = TaxonomyQuery(taxonomy2)

    stats1 = query1.get_statistics()
    stats2 = query2.get_statistics()

    print("Taxonomy Comparison:")
    print(f"Version 1: {stats1['total_nodes']} nodes, {stats1['leaf_nodes']} leaves")
    print(f"Version 2: {stats2['total_nodes']} nodes, {stats2['leaf_nodes']} leaves")

# CLI interface for taxonomy operations
def main_cli():
    """Command-line interface for taxonomy operations"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Load and query taxonomy trees'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Load command
    load_parser = subparsers.add_parser('load', help='Load and display taxonomy')
    load_parser.add_argument('--json', type=str, required=True,
                             help='Path to taxonomy JSON file')
    load_parser.add_argument('--assignments', type=str,
                             help='Path to assignments TSV file (optional)')
    load_parser.add_argument('--max-depth', type=int,
                             help='Maximum depth to display')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query taxonomy')
    query_parser.add_argument('--json', type=str, required=True,
                              help='Path to taxonomy JSON file')
    query_parser.add_argument('--assignments', type=str,
                              help='Path to assignments TSV file')
    query_parser.add_argument('--find', type=str,
                              help='Find concept by name')
    query_parser.add_argument('--search', type=str,
                              help='Search concepts by keyword')
    query_parser.add_argument('--stats', action='store_true',
                              help='Show taxonomy statistics')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert taxonomy format')
    convert_parser.add_argument('--input', type=str, required=True,
                                help='Input file path')
    convert_parser.add_argument('--input-format', type=str,
                                choices=['json', 'pickle'],
                                required=True, help='Input format')
    convert_parser.add_argument('--output', type=str, required=True,
                                help='Output file path')
    convert_parser.add_argument('--output-format', type=str,
                                choices=['json', 'pickle', 'dot'],
                                required=True, help='Output format')
    convert_parser.add_argument('--assignments', type=str,
                                help='Assignments file (for JSON input)')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize taxonomy')
    viz_parser.add_argument('--json', type=str, required=True,
                            help='Path to taxonomy JSON file')
    viz_parser.add_argument('--output', type=str, required=True,
                            help='Output file path (without extension)')
    viz_parser.add_argument('--format', type=str, default='pdf',
                            choices=['pdf', 'png', 'svg'],
                            help='Output format')

    args = parser.parse_args()

    if args.command == 'load':
        # Load and display taxonomy
        if args.assignments:
            root = TaxonomyLoader.load_with_assignments(args.json, args.assignments)
        else:
            root = TaxonomyLoader.from_json(args.json)

        TaxonomyVisualizer.print_tree(root, max_depth=args.max_depth)

    elif args.command == 'query':
        # Load taxonomy
        if args.assignments:
            root = TaxonomyLoader.load_with_assignments(args.json, args.assignments)
        else:
            root = TaxonomyLoader.from_json(args.json)

        query = TaxonomyQuery(root)

        if args.stats:
            stats = query.get_statistics()
            print("\nTaxonomy Statistics:")
            print(f"  Total nodes: {stats['total_nodes']}")
            print(f"  Leaf nodes: {stats['leaf_nodes']}")
            print(f"  Total documents: {stats['total_documents']}")
            print(f"  Max depth: {stats['max_depth']}")
            print(f"\nDepth distribution:")
            for depth in sorted(stats['depth_distribution'].keys()):
                print(f"    Depth {depth}: {stats['depth_distribution'][depth]} nodes")

        if args.find:
            node = query.find_concept(args.find)
            if node:
                print(f"\nFound concept: {node.name}")
                print(f"  Definition: {node.definition}")
                print(f"  Path: {query.get_path(node)}")
                print(f"  Documents: {len(node.doc_indices)}")
                print(f"  Children: {len(node.children)}")

                if node.children:
                    print(f"\n  Child concepts:")
                    for child in node.children:
                        print(f"    - {child.name} ({len(child.doc_indices)} docs)")
            else:
                print(f"Concept '{args.find}' not found")

        if args.search:
            matches = query.search_by_keyword(args.search)
            print(f"\nFound {len(matches)} concepts containing '{args.search}':")
            for node in matches:
                print(f"  - {query.get_path(node)}")
                print(f"    Definition: {node.definition}")
                print(f"    Documents: {len(node.doc_indices)}")

    elif args.command == 'convert':
        # Load taxonomy
        if args.input_format == 'json':
            if args.assignments:
                root = TaxonomyLoader.load_with_assignments(args.input, args.assignments)
            else:
                root = TaxonomyLoader.from_json(args.input)
        else:  # pickle
            root = TaxonomyLoader.from_pickle(args.input)

        # Save in new format
        if args.output_format == 'json':
            TaxonomySaver.to_json_with_docs(root, args.output)
            print(f"Saved taxonomy to JSON: {args.output}")
        elif args.output_format == 'pickle':
            TaxonomySaver.to_pickle(root, args.output)
            print(f"Saved taxonomy to pickle: {args.output}")
        elif args.output_format == 'dot':
            TaxonomyVisualizer.to_graphviz(root, args.output, format='pdf')
            print(f"Saved taxonomy visualization: {args.output}.pdf")

    elif args.command == 'visualize':
        root = TaxonomyLoader.from_json(args.json)
        TaxonomyVisualizer.to_graphviz(root, args.output, format=args.format)
        print(f"Visualization saved: {args.output}.{args.format}")

    else:
        parser.print_help()

if __name__ == '__main__':
    # Example usage
    # print("Taxonomy Loader Examples\n")

    # Uncomment to run examples
    # example_load_and_query()
    # example_modify_and_save()
    # example_compare_taxonomies()

    # Run CLI
    main_cli()
