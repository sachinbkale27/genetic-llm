"""
Upload the genetics Q&A dataset to HuggingFace Hub.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def load_and_upload(
    train_path: Path,
    val_path: Path,
    repo_id: str,
    private: bool = False
):
    """Load local JSON files and upload to HuggingFace."""

    print(f"Loading training data from {train_path}...")
    with open(train_path) as f:
        train_data = json.load(f)

    print(f"Loading validation data from {val_path}...")
    with open(val_path) as f:
        val_data = json.load(f)

    # Flatten the messages format for easier use
    def flatten_sample(sample):
        messages = sample["messages"]
        return {
            "system": messages[0]["content"],
            "question": messages[1]["content"],
            "answer": messages[2]["content"],
            "messages": messages  # Keep original format too
        }

    train_flat = [flatten_sample(s) for s in train_data]
    val_flat = [flatten_sample(s) for s in val_data]

    print(f"Creating HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_flat)
    val_dataset = Dataset.from_list(val_flat)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    print(f"Uploading to {repo_id}...")
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        commit_message="Upload genetics Q&A dataset"
    )

    print(f"\n✓ Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
    return dataset_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo ID (username/dataset-name)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    args = parser.parse_args()

    data_dir = Path(__file__).parent
    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"

    load_and_upload(train_path, val_path, args.repo, args.private)
