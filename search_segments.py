#!/usr/bin/env python3
"""
Search over timestamped lecture segments.

- Load index_segments.json
- For each user question:
    - embed the question
    - compute cosine similarity to all segments
    - show top-k segments with lecture_id + timestamp + snippet
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

from openai import OpenAI


def load_index(path: Path) -> List[Dict[str, Any]]:
    """Load segment index from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Index file must contain a list of segment records.")
    # Filter out segments without embeddings
    data = [seg for seg in data if "embedding" in seg]
    if not data:
        raise RuntimeError("No segments with embeddings found in index file.")
    return data


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def embed_question(question: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for the user question."""
    client = OpenAI()
    resp = client.embeddings.create(
        model=model,
        input=question,
    )
    return resp.data[0].embedding


def search_segments(
    question: str,
    segments: List[Dict[str, Any]],
    top_k: int = 5,
    model: str = "text-embedding-3-small",
) -> List[Tuple[float, Dict[str, Any]]]:
    """Return top_k most similar segments for the question."""
    q_emb = embed_question(question, model=model)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for seg in segments:
        emb = seg.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(q_emb, emb)
        scored.append((score, seg))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search over timestamped lecture segments."
    )
    parser.add_argument(
        "--index",
        type=str,
        default="index_segments.json",
        help="Path to segment index JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many segments to show per question.",
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    segments = load_index(index_path)

    print(f"[OK] Loaded {len(segments)} segments from {index_path}")
    print("Ask a question about the course. Empty line to exit.\n")

    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            print("Bye!")
            break

        print("\nSearching for relevant segments...\n")
        results = search_segments(
            q,
            segments,
            top_k=args.top_k,
            model=args.model,
        )

        if not results:
            print("No matching segments found.\n")
            continue

        for rank, (score, seg) in enumerate(results, start=1):
            lec_id = seg.get("lecture_id", "unknown")
            title = seg.get("lecture_title", "")
            ts = seg.get("timestamp_str", "?")
            snippet = seg.get("text", "").replace("\n", " ")

            print(f"[{rank}] {lec_id} ({title}) @ {ts}")
            print(f"    similarity: {score:.3f}")
            print(f"    snippet: {snippet[:200]}...")
            print()

        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()