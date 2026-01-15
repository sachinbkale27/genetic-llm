"""
Preprocess genetic Q&A data into training format for fine-tuning.

Handles data from multiple sources and outputs unified train.json and val.json.
"""

import json
import random
from pathlib import Path
from typing import Optional


SYSTEM_PROMPT = """You are a genetic research assistant with expertise in molecular biology, genomics, and genetic analysis. Provide accurate, scientifically-grounded answers to questions about genetics, DNA, gene expression, mutations, and related topics. Base your answers on established scientific knowledge and research."""


def format_for_training(
    question: str,
    context: str,
    answer: str,
) -> dict:
    """Format a single Q&A pair for instruction fine-tuning."""
    if context and context.strip():
        user_content = f"Context: {context}\n\nQuestion: {question}"
    else:
        user_content = question

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
    }


def load_all_sources(data_dir: Path) -> list[dict]:
    """Load data from all available source files."""
    all_samples = []

    # Try the combined file first (from download_all.py)
    combined_path = data_dir / "all_genetics.json"
    if combined_path.exists():
        print(f"Loading from {combined_path}")
        with open(combined_path) as f:
            data = json.load(f)
            for sample in data:
                all_samples.append({
                    "question": sample["question"],
                    "context": sample.get("context", ""),
                    "answer": sample["answer"],
                    "source": sample.get("source", "unknown")
                })
        return all_samples

    # Fallback: load individual source files
    source_files = [
        ("pubmedqa_genetics.json", "pubmedqa"),
    ]

    for filename, source in source_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"Loading from {filepath}")
            with open(filepath) as f:
                data = json.load(f)
                for sample in data:
                    # Handle different field names
                    answer = sample.get("answer") or sample.get("long_answer", "")
                    all_samples.append({
                        "question": sample["question"],
                        "context": sample.get("context", ""),
                        "answer": answer,
                        "source": source
                    })

    return all_samples


def deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicate questions."""
    seen = set()
    unique = []

    for sample in samples:
        # Normalize question for comparison
        q_normalized = sample["question"].lower().strip()[:200]
        if q_normalized not in seen:
            seen.add(q_normalized)
            unique.append(sample)

    return unique


def preprocess_dataset(
    data_dir: Optional[Path] = None,
    train_ratio: float = 0.9,
    shuffle_seed: int = 42
):
    """Convert raw dataset to training format."""
    if data_dir is None:
        data_dir = Path(__file__).parent

    # Load all sources
    raw_samples = load_all_sources(data_dir)

    if not raw_samples:
        print("No data found. Run download_all.py first:")
        print("  python data/download_all.py")
        return

    # Deduplicate
    raw_samples = deduplicate(raw_samples)

    print(f"Loaded {len(raw_samples)} unique samples")

    # Show source distribution
    sources = {}
    for s in raw_samples:
        src = s.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\nBy source:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    # Filter out samples with empty answers
    valid_samples = [s for s in raw_samples if s["answer"].strip()]
    print(f"\nValid samples (non-empty answers): {len(valid_samples)}")

    # Format for training
    training_data = []
    for sample in valid_samples:
        formatted = format_for_training(
            sample["question"],
            sample["context"],
            sample["answer"]
        )
        training_data.append(formatted)

    # Shuffle and split
    random.seed(shuffle_seed)
    random.shuffle(training_data)

    split_idx = int(len(training_data) * train_ratio)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    # Save
    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"\n✓ Training samples: {len(train_data)} → {train_path}")
    print(f"✓ Validation samples: {len(val_data)} → {val_path}")


if __name__ == "__main__":
    preprocess_dataset()
