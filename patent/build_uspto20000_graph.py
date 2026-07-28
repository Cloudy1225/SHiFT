#!/usr/bin/env python3
"""
Build a patent citation graph from USPTO20000.

Input Format
------------
First line:
    A Python dictionary representing patent citation relationships, for example:
    {'9532847': {'6889839', '6460718'}, ...}

Each subsequent line:
    patent_id<TAB>title. abstract<TAB>claim

Where:
- claim is ignored;
- title and abstract are split using the first ".";
- only patents that appear in the citation graph are retained;
- reissue patents (RE prefix, such as RE46551) are removed during parsing;
- citation edge direction:
    citing patent -> cited patent

Output Files
------------
nodes.tsv
edges.tsv
citation_map.json
stats.json
graph_data.pt (generated when PyTorch is installed)
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple


PatentText = Tuple[str, str]


def is_reissue_patent(patent_id: str) -> bool:
    """Return True if the ID is a reissue patent (RE prefix) and should be removed from the graph."""
    return patent_id.upper().startswith("RE")


def patent_sort_key(patent_id: str):
    """Sort patent numbers numerically."""
    if patent_id.isdigit():
        return 0, int(patent_id)
    return 1, patent_id


def clean_text(text: str) -> str:
    """Clean HTML escape characters and leading/trailing whitespace."""
    return html.unescape(text).strip()


def split_title_abstract(text: str) -> PatentText:
    """
    Split the title and abstract using the first period.

    Example:
        tornado safe room. a relocateable shelter ...

    Returns:
        title = tornado safe room
        abstract = a relocateable shelter ...
    """
    text = clean_text(text)

    title, separator, abstract = text.partition(".")

    if not separator:
        return title.strip(), ""

    return title.strip(), abstract.strip()


def parse_citation_map(line: str) -> Dict[str, Set[str]]:
    """Parse the patent citation dictionary on the first line of the file."""
    try:
        raw_map = ast.literal_eval(line.strip())
    except (SyntaxError, ValueError) as error:
        raise ValueError("The first line is not a valid Python dictionary.") from error

    if not isinstance(raw_map, dict):
        raise TypeError("The first line must be a dictionary.")

    citation_map: Dict[str, Set[str]] = {}

    for source, destinations in raw_map.items():
        source_id = str(source).strip()

        if not source_id:
            continue

        # Skip reissue patents (RE prefix), as reliable CPC labels cannot be obtained
        if is_reissue_patent(source_id):
            continue

        if destinations is None:
            destination_ids = set()

        elif isinstance(destinations, (set, list, tuple)):
            destination_ids = {
                str(destination).strip()
                for destination in destinations
                if str(destination).strip()
                and not is_reissue_patent(str(destination).strip())
            }

        else:
            raise TypeError(
                f"The citation list for patent {source_id} must be a set, list, or tuple."
            )

        # Remove self-citations
        destination_ids.discard(source_id)

        citation_map[source_id] = destination_ids

    return citation_map


def get_graph_nodes(citation_map: Dict[str, Set[str]]) -> Set[str]:
    """
    Graph nodes are the union of all sources and destinations.
    """
    nodes = set(citation_map.keys())

    for destinations in citation_map.values():
        nodes.update(destinations)

    return nodes


def read_uspto20000(
    input_path: Path,
) -> tuple[Dict[str, Set[str]], Dict[str, PatentText]]:
    """
    Read USPTO20000.

    Expected node record format:
        patent_id<TAB>title. abstract<TAB>claim

    The claim field is ignored directly.
    """
    patent_records: Dict[str, PatentText] = {}

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        first_line = file.readline()

        if not first_line:
            raise ValueError("The input file is empty.")

        citation_map = parse_citation_map(first_line)
        graph_nodes = get_graph_nodes(citation_map)

        for line_number, raw_line in enumerate(file, start=2):
            line = raw_line.rstrip("\r\n")

            if not line:
                continue

            # Split into at most three parts:
            # patent_id, title+abstract, claim
            fields = line.split("\t", maxsplit=2)

            if len(fields) < 2:
                print(f"Skipping malformed line {line_number}.")
                continue

            patent_id = fields[0].strip()

            # Remove patent records that are not in the graph
            if patent_id not in graph_nodes:
                continue

            title_abstract_text = fields[1]
            title, abstract = split_title_abstract(title_abstract_text)

            current_record = (title, abstract)
            previous_record = patent_records.get(patent_id)

            # If duplicate records exist, retain the one with more complete text
            if (
                previous_record is None
                or len(title) + len(abstract)
                > len(previous_record[0]) + len(previous_record[1])
            ):
                patent_records[patent_id] = current_record

    return citation_map, patent_records


def build_edges(
    citation_map: Dict[str, Set[str]],
) -> list[tuple[str, str]]:
    """
    Build deduplicated directed edges.

    Edge direction:
        citing patent -> cited patent
    """
    edges = {
        (source, destination)
        for source, destinations in citation_map.items()
        for destination in destinations
        if source != destination
    }

    return sorted(
        edges,
        key=lambda edge: (
            patent_sort_key(edge[0]),
            patent_sort_key(edge[1]),
        ),
    )


def write_graph(
    output_dir: Path,
    citation_map: Dict[str, Set[str]],
    patent_records: Dict[str, PatentText],
) -> dict:
    """Output the node table, edge table, and PyTorch graph data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_nodes = get_graph_nodes(citation_map)

    patent_ids = sorted(
        graph_nodes,
        key=patent_sort_key,
    )

    node_to_idx = {
        patent_id: index
        for index, patent_id in enumerate(patent_ids)
    }

    edges = build_edges(citation_map)

    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for source, destination in edges:
        out_degree[source] += 1
        in_degree[destination] += 1

    # ---------------------------------------------------------
    # nodes.tsv
    # ---------------------------------------------------------

    nodes_path = output_dir / "nodes.tsv"

    with nodes_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file, delimiter="\t")

        writer.writerow(
            [
                "node_idx",
                "patent_id",
                "title",
                "abstract",
                "in_degree",
                "out_degree",
            ]
        )

        for patent_id in patent_ids:
            title, abstract = patent_records.get(
                patent_id,
                ("", ""),
            )

            writer.writerow(
                [
                    node_to_idx[patent_id],
                    patent_id,
                    title,
                    abstract,
                    in_degree[patent_id],
                    out_degree[patent_id],
                ]
            )

    # ---------------------------------------------------------
    # edges.tsv
    # ---------------------------------------------------------

    edges_path = output_dir / "edges.tsv"

    with edges_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file, delimiter="\t")

        writer.writerow(
            [
                "src_idx",
                "dst_idx",
                "src_patent_id",
                "dst_patent_id",
            ]
        )

        for source, destination in edges:
            writer.writerow(
                [
                    node_to_idx[source],
                    node_to_idx[destination],
                    source,
                    destination,
                ]
            )

    # ---------------------------------------------------------
    # citation_map.json
    # ---------------------------------------------------------

    serializable_map = {
        source: sorted(
            destinations,
            key=patent_sort_key,
        )
        for source, destinations in sorted(
            citation_map.items(),
            key=lambda item: patent_sort_key(item[0]),
        )
    }

    citation_map_path = output_dir / "citation_map.json"

    citation_map_path.write_text(
        json.dumps(
            serializable_map,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    missing_text_patents = sorted(
        graph_nodes.difference(patent_records.keys()),
        key=patent_sort_key,
    )

    stats = {
        "num_nodes": len(patent_ids),
        "num_edges": len(edges),
        "num_patents_with_text": len(patent_records),
        "num_patents_without_text": len(missing_text_patents),
        "patents_without_text": missing_text_patents,
        "edge_direction": "citing_patent -> cited_patent",
        "text_fields": [
            "title",
            "abstract",
        ],
        "claim_included": False,
    }

    # ---------------------------------------------------------
    # graph_data.pt
    # ---------------------------------------------------------

    try:
        import torch

        if edges:
            edge_index = torch.tensor(
                [
                    [
                        node_to_idx[source]
                        for source, _ in edges
                    ],
                    [
                        node_to_idx[destination]
                        for _, destination in edges
                    ],
                ],
                dtype=torch.long,
            )

        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        titles = [
            patent_records.get(
                patent_id,
                ("", ""),
            )[0]
            for patent_id in patent_ids
        ]

        abstracts = [
            patent_records.get(
                patent_id,
                ("", ""),
            )[1]
            for patent_id in patent_ids
        ]

        graph_data = {
            "edge_index": edge_index,
            "patent_ids": patent_ids,
            "titles": titles,
            "abstracts": abstracts,
            "node_to_idx": node_to_idx,
            "edge_direction": "citing_patent -> cited_patent",
        }

        torch.save(
            graph_data,
            output_dir / "graph_data.pt",
        )

        stats["graph_data_pt_written"] = True

    except ImportError:
        stats["graph_data_pt_written"] = False
        stats["graph_data_pt_note"] = (
            "PyTorch is not installed, so graph_data.pt was not generated."
        )

    # ---------------------------------------------------------
    # stats.json
    # ---------------------------------------------------------

    stats_path = output_dir / "stats.json"

    stats_path.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a USPTO20000 patent citation graph."
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="USPTO20000 raw data file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("uspto20000_graph"),
        help="Output directory, default: uspto20000_graph",
    )

    args = parser.parse_args()

    citation_map, patent_records = read_uspto20000(
        args.input_file
    )

    stats = write_graph(
        output_dir=args.output_dir,
        citation_map=citation_map,
        patent_records=patent_records,
    )

    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nOutput directory: {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()