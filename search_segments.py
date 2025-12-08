#!/usr/bin/env python3
"""
Lecture-Grounded Learning Assistant.

This script:
- Loads segment-level lecture index from index_segments.json.
- For each user question:
    1) Uses embeddings to retrieve the top-k most relevant segments.
    2) Shows the matching segments (lecture, timestamp, snippet).
    3) Calls an LLM with a structured prompt:

        Prompt Template: Lecture-Grounded Learning Assistant
        Use only the retrieved lecture segments below to answer.
        Step through:
          1. Summarize the professor’s exact explanation
          2. Give a simple analogy
          3. Give a worked-out example
          4. Quiz me with 3 questions
          5. Point out common misconceptions
        If you are unsure, say so.

    4) Prints the model's teaching-style answer.

This turns your retrieval index into a small lecture-based tutor.
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

from openai import OpenAI


# ========================= Basic utilities =========================

def load_index(path: Path) -> List[Dict[str, Any]]:
    """Load segment index from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Index file must contain a list of segment records.")
    # Keep only segments that already have embeddings
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


def embed_text(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for a text string."""
    client = OpenAI()
    resp = client.embeddings.create(
        model=model,
        input=text,
    )
    return resp.data[0].embedding


def search_segments(
    question: str,
    segments: List[Dict[str, Any]],
    top_k: int,
    embed_model: str = "text-embedding-3-small",
) -> List[Tuple[float, Dict[str, Any]]]:
    """Return top_k most similar segments for the question."""
    q_emb = embed_text(question, model=embed_model)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for seg in segments:
        emb = seg.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(q_emb, emb)
        scored.append((score, seg))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ========================= LLM teaching assistant =========================

def build_context_from_segments(
    results: List[Tuple[float, Dict[str, Any]]],
    max_chars_per_segment: int = 600,
) -> str:
    """
    Build a context string from retrieved segments for the LLM.

    Each segment includes:
      - lecture id and title
      - timestamp
      - truncated transcript text
    """
    lines: List[str] = []
    for rank, (score, seg) in enumerate(results, start=1):
        lec_id = seg.get("lecture_id", "unknown")
        title = seg.get("lecture_title", "")
        ts = seg.get("timestamp_str", "?")
        text = seg.get("text", "").replace("\n", " ")
        if len(text) > max_chars_per_segment:
            text = text[:max_chars_per_segment] + " ..."
        lines.append(
            f"[SEGMENT {rank}] lecture_id={lec_id}, title={title}, timestamp={ts}, "
            f"similarity={score:.3f}\n{text}"
        )
    return "\n\n".join(lines)


def lecture_grounded_answer(
    question: str,
    retrieved: List[Tuple[float, Dict[str, Any]]],
    chat_model: str = "gpt-4.1-mini",
) -> str:
    """
    Call an LLM to answer the question using only the retrieved lecture segments.

    The prompt enforces the 5 teaching steps:
      1. Summarize the professor’s explanation
      2. Simple analogy
      3. Worked example
      4. Quiz with 3 questions
      5. Common misconceptions
    """
    if not retrieved:
        return (
            "I could not find any relevant lecture segments for this question. "
            "You may want to try a different phrasing or check the transcripts."
        )

    context = build_context_from_segments(retrieved)

    system_msg = (
        "You are a lecture-grounded teaching assistant for an EECS course. "
        "You must ONLY use the provided lecture transcript segments as your source. "
        "If something is not in the segments, say that you are unsure.\n"
        "Your goal is to help the student deeply understand the concept "
        "as the professor explained it in lecture."
    )

    user_msg = f"""
Below are the retrieved lecture segments related to the student's question:

{context}

Student question:
{question}

Use ONLY these lecture segments to answer.

Step through in order:

1. Summarize the professor’s exact explanation from these segments. 
   Refer to lecture and timestamp when helpful (e.g., "In Lec4 at 22:14, the professor says...").
2. Give a simple analogy that matches the lecture's spirit (and say if the analogy is your own).
3. Give a worked-out example (using numbers or a concrete scenario) that is consistent with the lecture.
4. Quiz the student with 3 short questions to check their understanding.
5. Point out 2–3 common misconceptions related to this topic and clarify them.

If the segments do not contain enough information to fully answer, clearly say what is missing.
"""

    client = OpenAI()
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )

    return resp.choices[0].message.content.strip()


# ========================= Main interactive loop =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lecture-grounded learning assistant over timestamped lecture segments."
    )
    parser.add_argument(
        "--index",
        type=str,
        default="index_segments.json",
        help="Path to segment index JSON file.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI chat model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many segments to retrieve per question.",
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    segments = load_index(index_path)

    print(f"[OK] Loaded {len(segments)} segments from {index_path}")
    print("Lecture-Grounded Learning Assistant")
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

        print("\n[Search] Finding relevant lecture segments...\n")
        results = search_segments(
            q,
            segments,
            top_k=args.top_k,
            embed_model=args.embed_model,
        )

        if not results:
            print("No matching segments found.\n")
            continue

        # Show retrieved segments for transparency
        print("Top matching segments:")
        for rank, (score, seg) in enumerate(results, start=1):
            lec_id = seg.get("lecture_id", "unknown")
            title = seg.get("lecture_title", "")
            ts = seg.get("timestamp_str", "?")
            snippet = seg.get("text", "").replace("\n", " ")
            print(f"[{rank}] {lec_id} ({title}) @ {ts}")
            print(f"    similarity: {score:.3f}")
            print(f"    snippet: {snippet[:200]}...")
            print()

        print("-" * 60)
        print("\n[Assistant] Generating lecture-grounded explanation...\n")

        answer = lecture_grounded_answer(
            question=q,
            retrieved=results,
            chat_model=args.chat_model,
        )

        print(answer)
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
