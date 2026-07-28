#!/usr/bin/env python3
"""
Encode patent title+abstract into node feature vectors using a pretrained LM.

The encoded features replace the placeholder `x` in patent_cpc_dataset.pt.

Usage:
    python encode_patent_features.py [--encoder roberta] [--batch_size 32]

Output:
    patent/{encoder_name}_patent_features.pt   — FloatTensor (N, hidden_dim)

Supported encoders  (must match MODEL_PATHs keys in SHiFT/utils.py):
    MiniLM, SentenceBert, e5-large, roberta (default)
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PATENT_DIR = Path(__file__).parent
SHIFT_DIR  = PATENT_DIR.parent
sys.path.insert(0, str(SHIFT_DIR))

from lm import TextEncoder  # noqa: E402  (found via sys.path above)


# ---------------------------------------------------------------------------
# Pooling strategy per encoder family
# ---------------------------------------------------------------------------
# sentence-transformers models are trained with mean pooling;
# plain encoder models (BERT-style) use CLS token pooling.
POOLING_STRATEGY = {
    "MiniLM":       "mean",
    "SentenceBert": "mean",
    "e5-large":     "mean",
    "roberta":      "mean",
}


def build_input_texts(texts: list[tuple[str, str]]) -> list[str]:
    """
    Concatenate title and abstract into a single string.
    Falls back to 'Empty text' when both fields are blank.
    """
    result = []
    for title, abstract in texts:
        combined = f"{title} {abstract}".strip()
        result.append(combined if combined else "Empty text")
    return result


def encode_features(
    encoder_name: str,
    batch_size: int,
    device: str,
    max_length: int,
) -> torch.Tensor:
    dataset_path = PATENT_DIR / "patent_cpc_dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run build_patent_cpc_dataset.py first."
        )

    print("Loading dataset...")
    dataset     = torch.load(dataset_path, weights_only=False)
    input_texts = build_input_texts(dataset["texts"])
    n           = len(input_texts)
    print(f"Encoding {n} patents with [{encoder_name}] on {device} ...")

    pooling  = POOLING_STRATEGY.get(encoder_name, "cls")
    encoder  = TextEncoder(encoder_name, encoder_type="LM", device=device)
    encoder.model.eval()

    all_embs: list[torch.Tensor] = []

    for start in tqdm(range(0, n, batch_size), desc=f"[{encoder_name}]"):
        batch = input_texts[start : start + batch_size]
        with torch.no_grad():
            emb = encoder.forward(batch, pooling=pooling, max_length=max_length)
        all_embs.append(emb.cpu().float())

    features = torch.cat(all_embs, dim=0)  # (N, hidden_dim)
    assert features.shape[0] == n, f"Shape mismatch: {features.shape[0]} != {n}"
    print(f"Feature tensor shape: {features.shape}")

    out_path = PATENT_DIR / f"{encoder_name}_patent_features.pt"
    torch.save(features, out_path)
    print(f"Saved → {out_path}")
    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode USPTO patent text into node feature vectors."
    )
    parser.add_argument(
        "--encoder", type=str, default="roberta",
        choices=["MiniLM", "SentenceBert", "e5-large", "roberta"],
        help="Pretrained LM encoder to use (default: roberta)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Encoding batch size (default: 32; lower if OOM)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda:0 if available)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Max token length (default: 512)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_features(
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )
