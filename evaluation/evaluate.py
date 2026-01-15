"""
Evaluate the fine-tuned GeneticLLM model.

Metrics:
- BLEU score (text similarity)
- ROUGE scores (recall-oriented)
- Genetic terminology accuracy
- Response relevance
"""

import json
import re
from pathlib import Path
from typing import Optional
from collections import Counter

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Install evaluation deps: pip install nltk rouge-score")


GENETIC_TERMS = {
    "gene", "dna", "rna", "genome", "chromosome", "mutation", "allele",
    "nucleotide", "protein", "expression", "transcription", "translation",
    "codon", "exon", "intron", "promoter", "enhancer", "methylation",
    "epigenetic", "genotype", "phenotype", "polymorphism", "snp", "variant",
    "crispr", "cas9", "sequencing", "pcr", "primer", "replication",
    "helicase", "polymerase", "ligase", "ribosome", "mrna", "trna",
    "splicing", "histone", "chromatin", "telomere", "centromere"
}


def extract_genetic_terms(text: str) -> set:
    """Extract recognized genetic terms from text."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return words & GENETIC_TERMS


def calculate_terminology_score(prediction: str, reference: str) -> dict:
    """Calculate genetic terminology coverage."""
    pred_terms = extract_genetic_terms(prediction)
    ref_terms = extract_genetic_terms(reference)

    if not ref_terms:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    precision = len(pred_terms & ref_terms) / len(pred_terms) if pred_terms else 0
    recall = len(pred_terms & ref_terms) / len(ref_terms)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_terms": list(pred_terms),
        "ref_terms": list(ref_terms),
        "matched": list(pred_terms & ref_terms)
    }


def calculate_bleu(prediction: str, reference: str) -> float:
    """Calculate BLEU score."""
    if not METRICS_AVAILABLE:
        return 0.0

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)


def calculate_rouge(prediction: str, reference: str) -> dict:
    """Calculate ROUGE scores."""
    if not METRICS_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)

    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }


def evaluate_response(prediction: str, reference: str) -> dict:
    """Evaluate a single prediction against reference."""
    return {
        "bleu": calculate_bleu(prediction, reference),
        "rouge": calculate_rouge(prediction, reference),
        "terminology": calculate_terminology_score(prediction, reference),
        "length_ratio": len(prediction) / len(reference) if reference else 0
    }


def evaluate_dataset(
    predictions: list[str],
    references: list[str],
    questions: Optional[list[str]] = None
) -> dict:
    """Evaluate all predictions and compute aggregate metrics."""
    results = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        eval_result = evaluate_response(pred, ref)
        if questions:
            eval_result["question"] = questions[i]
        results.append(eval_result)

    # Aggregate metrics
    n = len(results)
    aggregate = {
        "num_samples": n,
        "avg_bleu": sum(r["bleu"] for r in results) / n,
        "avg_rouge1": sum(r["rouge"]["rouge1"] for r in results) / n,
        "avg_rouge2": sum(r["rouge"]["rouge2"] for r in results) / n,
        "avg_rougeL": sum(r["rouge"]["rougeL"] for r in results) / n,
        "avg_terminology_f1": sum(r["terminology"]["f1"] for r in results) / n,
        "avg_length_ratio": sum(r["length_ratio"] for r in results) / n,
    }

    return {
        "aggregate": aggregate,
        "individual": results
    }


def run_evaluation(
    val_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None
):
    """Run evaluation on validation set."""
    if val_path is None:
        val_path = Path(__file__).parent.parent / "data" / "val.json"

    if not val_path.exists():
        print(f"Validation file not found: {val_path}")
        print("Run data preprocessing first.")
        return

    with open(val_path) as f:
        val_data = json.load(f)

    # Extract questions and reference answers
    questions = []
    references = []

    for sample in val_data:
        messages = sample.get("messages", [])
        for msg in messages:
            if msg["role"] == "user":
                questions.append(msg["content"])
            elif msg["role"] == "assistant":
                references.append(msg["content"])

    print(f"Loaded {len(questions)} validation samples")

    # If predictions file provided, load and evaluate
    if predictions_path and predictions_path.exists():
        with open(predictions_path) as f:
            predictions = json.load(f)

        results = evaluate_dataset(predictions, references, questions)

        print("\n=== Evaluation Results ===")
        print(f"Samples: {results['aggregate']['num_samples']}")
        print(f"BLEU: {results['aggregate']['avg_bleu']:.4f}")
        print(f"ROUGE-1: {results['aggregate']['avg_rouge1']:.4f}")
        print(f"ROUGE-2: {results['aggregate']['avg_rouge2']:.4f}")
        print(f"ROUGE-L: {results['aggregate']['avg_rougeL']:.4f}")
        print(f"Terminology F1: {results['aggregate']['avg_terminology_f1']:.4f}")

        # Save results
        output_path = Path(__file__).parent / "eval_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_path}")

        return results

    else:
        print("\nNo predictions file provided.")
        print("To evaluate, generate predictions using inference/query.py")
        print("Then run: python evaluate.py --predictions predictions.json")

        return {"questions": questions, "references": references}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, help="Path to validation data")
    parser.add_argument("--predictions", type=Path, help="Path to model predictions")
    args = parser.parse_args()

    run_evaluation(args.val, args.predictions)
