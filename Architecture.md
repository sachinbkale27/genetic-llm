# GeneticLLM

End-to-end LLM fine-tuning for domain-specific Q&A in genetic research.

## Architecture

![GeneticLLM Architecture](docs/architecture.png)

## What It Does

Fine-tunes a small language model (Qwen2-1.5B) to answer questions about genetics, molecular biology, and genomics — topics like CRISPR, DNA methylation, gene expression, mutations, and inheritance.

## Path for Implementation

| Path                   | Implementation |
|------------------------|----------------|
| **Data Engineering**   | Multi-source pipeline aggregating 89k samples from PubMedQA, MMLU, MedMCQA, SciQ |
| **Efficient Training** | QLoRA (4-bit quantization + LoRA) — runs on free Colab T4 GPU |
| **Domain Adaptation**  | Filtering and preprocessing for genetics-specific terminology |
| **MLOps**              | Dataset hosted on HuggingFace Hub, reproducible training notebook |
| **Evaluation**         | BLEU, ROUGE, and domain-specific terminology metrics |

## Project Structure

```
genetic-llm/
├── data/
│   ├── download_all.py      # Fetches from 4 sources
│   ├── preprocess.py        # Formats for instruction tuning
│   └── upload_to_hf.py      # Pushes to HuggingFace Hub
├── training/
│   └── finetune.ipynb       # One-click Colab training
├── evaluation/
│   └── evaluate.py          # Metrics pipeline
├── inference/
│   └── query.py             # Interactive & batch inference
└── README.md
```

## Training Dashboard

![Training Dashboard](docs/genetic-llm-training-dashboard.png)

Training metrics tracked via Weights & Biases showing loss convergence, learning rate schedule, and GPU utilization during fine-tuning on A100.

## Tech Stack

- **Base Model**: Qwen2-1.5B-Instruct (Apache 2.0)
- **Training**: Unsloth + QLoRA + TRL's SFTTrainer
- **Data**: HuggingFace Datasets
- **Tracking**: Weights & Biases
- **Infra**: Google Colab (A100/T4)

## Links

- **Model**: https://huggingface.co/sachinbkale27/genetics-llm-lora-v1
- **Dataset**: https://huggingface.co/datasets/sachinbkale27/genetics-qa
- **GitHub**: https://github.com/sachinbkale27/genetic-llm
