# GeneticLLM

A fine-tuned large language model specialized for genetic research Q&A. Built using QLoRA (4-bit quantization + LoRA) for efficient training on consumer hardware.

## Overview

This project demonstrates:
- **Dataset Curation**: Filtering PubMedQA for genetics-specific Q&A pairs
- **Efficient Fine-tuning**: QLoRA with Unsloth for 4-bit training on free Colab GPUs
- **Domain Adaptation**: Specializing a general LLM for genetic research terminology
- **Evaluation Pipeline**: Measuring BLEU, ROUGE, and domain-specific terminology accuracy

## Links

- **Model**: [sachinbkale27/genetics-llm-lora-v1](https://huggingface.co/sachinbkale27/genetics-llm-lora-v1)
- **Dataset**: [sachinbkale27/genetics-qa](https://huggingface.co/datasets/sachinbkale27/genetics-qa)
- **GitHub**: [sachinbkale27/genetic-llm](https://github.com/sachinbkale27/genetic-llm)

## Project Structure

```
genetic-llm/
├── data/
│   ├── download_pubmedqa.py   # Download and filter PubMedQA
│   └── preprocess.py          # Convert to training format
├── training/
│   └── finetune.ipynb         # Colab notebook for QLoRA fine-tuning
├── evaluation/
│   └── evaluate.py            # Evaluation metrics and testing
├── inference/
│   └── query.py               # Query the fine-tuned model
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Prepare Data

```bash
# Install dependencies
pip install -r requirements.txt

# Download and filter PubMedQA for genetics
python data/download_pubmedqa.py

# Convert to training format
python data/preprocess.py
```

### 2. Fine-tune on Google Colab

1. Open `training/finetune.ipynb` in [Google Colab](https://colab.research.google.com)
2. Select **Runtime > Change runtime type > T4 GPU**
3. Upload `data/train.json` when prompted
4. Run all cells to fine-tune
5. Download the LoRA adapters

### 3. Run Inference

```python
# Load from HuggingFace
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "sachinbkale27/genetics-llm-lora-v1",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Ask a question
messages = [
    {"role": "system", "content": "You are a genetic research assistant."},
    {"role": "user", "content": "What is CRISPR-Cas9?"}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Or use the CLI:

```bash
# With fine-tuned adapters
python inference/query.py --adapter sachinbkale27/genetics-llm-lora-v1

# Single question
python inference/query.py --adapter sachinbkale27/genetics-llm-lora-v1 \
    --question "What is the role of CRISPR-Cas9 in gene editing?"
```

### 4. Evaluate

```bash
# Generate predictions on validation set
python inference/query.py --adapter path/to/genetic-llm-lora \
    --batch data/val.json --output predictions.json

# Run evaluation
python evaluation/evaluate.py --predictions predictions.json
```

## Technical Details

### Base Model
- **Qwen2-1.5B-Instruct**: Small, capable model with Apache 2.0 license
- Chosen for: Good instruction-following, fits in free Colab GPU memory

### Training Configuration
- **Quantization**: 4-bit NormalFloat (NF4) via bitsandbytes
- **LoRA Rank**: 16 (balance between capacity and memory)
- **Target Modules**: All attention and MLP projections
- **Learning Rate**: 2e-4 with linear decay
- **Epochs**: 3

### Evaluation Metrics
- **BLEU**: N-gram precision for text similarity
- **ROUGE-1/2/L**: Recall-oriented summary evaluation
- **Terminology F1**: Coverage of genetic research terms

## Dataset

The training data is derived from [PubMedQA](https://pubmedqa.github.io/), filtered for genetics-related content using domain-specific keywords:

- Gene expression and regulation
- DNA/RNA structure and function
- Mutations and variants
- Sequencing technologies
- CRISPR and gene editing
- Epigenetics

## Results

| Metric | Base Model | Fine-tuned |
|--------|------------|------------|
| BLEU | TBD | TBD |
| ROUGE-L | TBD | TBD |
| Terminology F1 | TBD | TBD |

*Fill in after training*

## Example Outputs

**Question**: What is the role of CRISPR-Cas9 in gene editing?

**Response**: *[Add after fine-tuning]*

---

**Question**: How do single nucleotide polymorphisms affect disease risk?

**Response**: *[Add after fine-tuning]*

## Future Improvements

- [ ] Add more training data from genetics textbooks
- [ ] Implement retrieval-augmented generation (RAG) with research papers
- [ ] Fine-tune larger models (Qwen2-7B, Llama-3-8B) with more compute
- [ ] Add citation generation for responses
- [ ] Build a web interface with Gradio

## License

MIT License. The base model (Qwen2) is Apache 2.0 licensed.

## Acknowledgments

- [PubMedQA](https://pubmedqa.github.io/) for the dataset
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Qwen Team](https://github.com/QwenLM/Qwen2) for the base model
