"""
Download multiple datasets for genetic research Q&A training.

Sources:
- PubMedQA (genetics-filtered)
- MMLU Medical Genetics
- MMLU College Biology
- MMLU High School Biology
- MedMCQA (genetics/biochemistry subjects)
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
    "recombination", "replication", "pcr", "gwas", "snp", "indel",
    "amino acid", "enzyme", "receptor", "ligand", "pathway", "cell",
    "nucleus", "cytoplasm", "membrane", "organelle", "mitosis", "meiosis"
]


def contains_genetic_terms(text: str) -> bool:
    """Check if text contains genetic research terminology."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in GENETIC_KEYWORDS)


def download_pubmedqa() -> list[dict]:
    """Download and filter PubMedQA for genetics."""
    print("\n[1/4] Downloading PubMedQA...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    samples = []
    for sample in dataset:
        question = sample.get("question", "")
        context = " ".join(sample.get("context", {}).get("contexts", []))
        long_answer = sample.get("long_answer", "")

        combined_text = f"{question} {context} {long_answer}"

        if contains_genetic_terms(combined_text):
            samples.append({
                "question": question,
                "context": context,
                "answer": long_answer,
                "source": "pubmedqa"
            })

    print(f"   PubMedQA: {len(samples)} genetics samples")
    return samples


def download_mmlu_genetics() -> list[dict]:
    """Download MMLU medical genetics, college biology, HS biology."""
    print("\n[2/4] Downloading MMLU subsets...")

    subsets = [
        ("medical_genetics", "cais/mmlu", "medical_genetics"),
        ("college_biology", "cais/mmlu", "college_biology"),
        ("high_school_biology", "cais/mmlu", "high_school_biology"),
    ]

    samples = []
    choice_labels = ["A", "B", "C", "D"]

    for name, dataset_id, config in subsets:
        try:
            # Load both test and validation splits
            for split in ["test", "validation"]:
                try:
                    dataset = load_dataset(dataset_id, config, split=split)

                    for sample in dataset:
                        question = sample["question"]
                        choices = sample["choices"]
                        answer_idx = sample["answer"]

                        # Format as Q&A with explanation
                        choices_text = "\n".join(
                            f"{choice_labels[i]}. {c}"
                            for i, c in enumerate(choices)
                        )
                        correct_answer = choices[answer_idx]

                        samples.append({
                            "question": f"{question}\n\nOptions:\n{choices_text}",
                            "context": "",
                            "answer": f"The correct answer is {choice_labels[answer_idx]}. {correct_answer}",
                            "source": f"mmlu_{name}"
                        })
                except Exception:
                    continue

            print(f"   MMLU {name}: loaded")
        except Exception as e:
            print(f"   MMLU {name}: skipped ({e})")

    print(f"   MMLU total: {len(samples)} samples")
    return samples


def download_medmcqa() -> list[dict]:
    """Download MedMCQA filtered for genetics-related subjects."""
    print("\n[3/4] Downloading MedMCQA...")

    # Subjects relevant to genetics
    genetics_subjects = {
        "Biochemistry", "Anatomy", "Physiology", "Pathology",
        "Microbiology", "Pharmacology", "Medicine"
    }

    samples = []
    choice_labels = ["A", "B", "C", "D"]

    try:
        dataset = load_dataset("openlifescienceai/medmcqa", split="train")

        for sample in dataset:
            subject = sample.get("subject_name", "")
            question = sample.get("question", "")

            # Filter by subject OR by keyword content
            if subject in genetics_subjects or contains_genetic_terms(question):
                choices = [
                    sample.get("opa", ""),
                    sample.get("opb", ""),
                    sample.get("opc", ""),
                    sample.get("opd", "")
                ]
                answer_idx = sample.get("cop", 0)
                explanation = sample.get("exp", "")

                if not all(choices) or answer_idx is None:
                    continue

                choices_text = "\n".join(
                    f"{choice_labels[i]}. {c}"
                    for i, c in enumerate(choices)
                )

                answer_text = f"The correct answer is {choice_labels[answer_idx]}. {choices[answer_idx]}"
                if explanation:
                    answer_text += f"\n\nExplanation: {explanation}"

                samples.append({
                    "question": f"{question}\n\nOptions:\n{choices_text}",
                    "context": "",
                    "answer": answer_text,
                    "source": "medmcqa"
                })

        print(f"   MedMCQA: {len(samples)} genetics-related samples")
    except Exception as e:
        print(f"   MedMCQA: failed ({e})")

    return samples


def download_sciq() -> list[dict]:
    """Download SciQ dataset filtered for biology/genetics."""
    print("\n[4/4] Downloading SciQ...")

    samples = []

    try:
        dataset = load_dataset("allenai/sciq", split="train")

        for sample in dataset:
            question = sample.get("question", "")
            support = sample.get("support", "")
            correct = sample.get("correct_answer", "")

            combined = f"{question} {support} {correct}"

            if contains_genetic_terms(combined):
                # Include distractors as context
                distractors = [
                    sample.get("distractor1", ""),
                    sample.get("distractor2", ""),
                    sample.get("distractor3", "")
                ]

                answer = correct
                if support:
                    answer = f"{correct}\n\nExplanation: {support}"

                samples.append({
                    "question": question,
                    "context": "",
                    "answer": answer,
                    "source": "sciq"
                })

        print(f"   SciQ: {len(samples)} genetics samples")
    except Exception as e:
        print(f"   SciQ: failed ({e})")

    return samples


def main():
    """Download all datasets and merge."""
    print("=" * 50)
    print("Downloading Genetics Q&A Training Data")
    print("=" * 50)

    all_samples = []

    # Download each source
    all_samples.extend(download_pubmedqa())
    all_samples.extend(download_mmlu_genetics())
    all_samples.extend(download_medmcqa())
    all_samples.extend(download_sciq())

    # Summary by source
    print("\n" + "=" * 50)
    print("Summary by source:")
    sources = {}
    for s in all_samples:
        src = s["source"]
        sources[src] = sources.get(src, 0) + 1

    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    print(f"\nTotal samples: {len(all_samples)}")

    # Save merged dataset
    output_path = Path(__file__).parent / "all_genetics.json"
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"Saved to {output_path}")

    return all_samples


if __name__ == "__main__":
    main()
