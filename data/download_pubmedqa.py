"""
Download and filter PubMedQA dataset for genetic research Q&A.
"""

import json
from pathlib import Path
from datasets import load_dataset


GENETIC_KEYWORDS = [
    "gene", "genetic", "genome", "genomic", "dna", "rna", "mutation",
    "allele", "chromosome", "hereditary", "inheritance", "polymorphism",
    "sequencing", "crispr", "transcription", "translation", "protein",
    "expression", "variant", "genotype", "phenotype", "epigenetic",
    "methylation", "histone", "nucleotide", "codon", "exon", "intron",
    "splicing", "promoter", "enhancer", "telomere", "mitochondrial",
    "recombination", "replication", "pcr", "gwas", "snp", "indel"
]


def contains_genetic_terms(text: str) -> bool:
    """Check if text contains genetic research terminology."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in GENETIC_KEYWORDS)


def download_and_filter():
    """Download PubMedQA and filter for genetics-related entries."""
    print("Downloading PubMedQA dataset...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    genetic_samples = []

    print("Filtering for genetics-related questions...")
    for sample in dataset:
        question = sample.get("question", "")
        context = " ".join(sample.get("context", {}).get("contexts", []))
        long_answer = sample.get("long_answer", "")

        combined_text = f"{question} {context} {long_answer}"

        if contains_genetic_terms(combined_text):
            genetic_samples.append({
                "question": question,
                "context": context,
                "long_answer": long_answer,
                "final_decision": sample.get("final_decision", "")
            })

    print(f"Found {len(genetic_samples)} genetics-related samples")

    output_path = Path(__file__).parent / "pubmedqa_genetics.json"
    with open(output_path, "w") as f:
        json.dump(genetic_samples, f, indent=2)

    print(f"Saved to {output_path}")
    return genetic_samples


if __name__ == "__main__":
    download_and_filter()
