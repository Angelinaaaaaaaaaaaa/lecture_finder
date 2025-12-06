#!/usr/bin/env python3
"""
Build segment-level index from timestamped lecture transcripts.

- Input:  timestamped .txt files under data/transcripts/
- Format example:

    1:35
    But again, I'm really sorry...

    1:44
    Um, the class is going to be recorded...

- Output: index_segments.json containing a list of segments:
    [
      {
        "lecture_id": "lec_01_intro",
        "lecture_title": "lec_01_intro",
        "timestamp_str": "1:35",
        "time_seconds": 95,
        "text": "...",
        "embedding": [ ... ]
      },
      ...
    ]
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI


TIMESTAMP_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")


def parse_timestamp_to_seconds(ts: str) -> Optional[int]:
    """Convert 'mm:ss' or 'hh:mm:ss' to total seconds."""
    m = TIMESTAMP_RE.match(ts)
    if not m:
        return None
    h_str, m_str, s_str = m.groups()
    h = 0 if s_str is None and m_str is not None and len(h_str) <= 2 and int(h_str) < 60 else int(h_str)
    # For simplicity, treat first group as minutes if no hours part given
    if s_str is None:
        minutes = int(h_str)
        seconds = int(m_str)
        return minutes * 60 + seconds
    else:
        # hh:mm:ss
        h = int(h_str)
        minutes = int(m_str)
        seconds = int(s_str)
        return h * 3600 + minutes * 60 + seconds


def is_timestamp_line(line: str) -> bool:
    """Return True if the line looks like a pure timestamp."""
    return TIMESTAMP_RE.match(line.strip()) is not None


def parse_transcript_file(path: Path) -> List[Dict[str, Any]]:
    """
    Parse one timestamped transcript file into segments.

    Each segment:
      - starts at a timestamp line
      - includes all following text lines until next timestamp
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    segments: List[Dict[str, Any]] = []
    current_ts: Optional[str] = None
    current_lines: List[str] = []

    def flush_current():
        """Flush current segment into segments list."""
        nonlocal current_ts, current_lines
        if current_ts is None:
            return
        content = "\n".join([l for l in current_lines if l.strip()])
        if not content.strip():
            # Empty segment, skip
            current_ts = None
            current_lines = []
            return
        seconds = parse_timestamp_to_seconds(current_ts)
        segments.append(
            {
                "timestamp_str": current_ts.strip(),
                "time_seconds": seconds if seconds is not None else None,
                "text": content,
            }
        )
        current_ts = None
        current_lines = []

    for line in lines:
        if is_timestamp_line(line):
            # New timestamp: flush previous segment, start a new one
            flush_current()
            current_ts = line.strip()
            current_lines = []
        else:
            # Normal content line
            if current_ts is not None:
                current_lines.append(line)

    # Flush the last segment
    flush_current()
    return segments


def load_all_transcripts(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all .txt transcripts and parse into segment records."""
    all_segments: List[Dict[str, Any]] = []

    for path in sorted(data_dir.glob("*.txt")):
        lecture_id = path.stem
        lecture_title = lecture_id.replace("_", " ").title()

        print(f"[Parse] {path}")

        segs = parse_transcript_file(path)
        print(f"  -> {len(segs)} segments")

        for seg in segs:
            seg["lecture_id"] = lecture_id
            seg["lecture_title"] = lecture_title
            seg["source_path"] = str(path)
            all_segments.append(seg)

    if not all_segments:
        raise RuntimeError(f"No .txt transcripts found in {data_dir}")
    return all_segments


def embed_segments(
    segments: List[Dict[str, Any]],
    model: str = "text-embedding-3-small",
) -> None:
    """Call OpenAI embeddings once per segment text."""
    client = OpenAI()

    for i, seg in enumerate(segments):
        text = seg["text"]
        # Optional: truncate very long segments
        max_chars = 8000
        if len(text) > max_chars:
            text_for_embedding = text[:max_chars]
        else:
            text_for_embedding = text

        print(f"[Embedding] {i+1}/{len(segments)}", end="\r")

        resp = client.embeddings.create(
            model=model,
            input=text_for_embedding,
        )
        seg["embedding"] = resp.data[0].embedding

    print(f"\n[OK] Embedded {len(segments)} segments.")


def save_index(segments: List[Dict[str, Any]], out_path: Path) -> None:
    """Save segment index as JSON."""
    out_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Index saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build segment-level index from timestamped lecture transcripts."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/transcripts",
        help="Directory containing timestamped .txt transcripts.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="index_segments.json",
        help="Path to save segment index JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {data_dir}")

    segments = load_all_transcripts(data_dir)
    embed_segments(segments, model=args.model)
    save_index(segments, Path(args.out))


if __name__ == "__main__":
    main()
