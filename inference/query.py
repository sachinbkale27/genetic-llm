"""
Query the fine-tuned GeneticLLM model.

Supports both local inference and batch processing for evaluation.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SYSTEM_PROMPT = """You are a genetic research assistant with expertise in molecular biology, genomics, and genetic analysis. Provide accurate, scientifically-grounded answers to questions about genetics, DNA, gene expression, mutations, and related topics."""


class GeneticLLM:
    """Wrapper for the fine-tuned genetic research model."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the model.

        Args:
            base_model: HuggingFace model ID for the base model
            adapter_path: Path to LoRA adapters (if fine-tuned)
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        print(f"Loading base model: {base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )

        # Load LoRA adapters if provided
        if adapter_path:
            print(f"Loading LoRA adapters from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print("Model ready!")

    def query(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Ask a genetics question.

        Args:
            question: The question to ask
            context: Optional context (e.g., from a research paper)
            max_tokens: Maximum response length
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            Model's response
        """
        # Build prompt
        if context:
            user_content = f"Context: {context}\n\nQuestion: {question}"
        else:
            user_content = question

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def batch_query(
        self,
        questions: list[str],
        contexts: Optional[list[str]] = None,
        **kwargs
    ) -> list[str]:
        """Process multiple questions."""
        if contexts is None:
            contexts = [""] * len(questions)

        responses = []
        for q, c in zip(questions, contexts):
            responses.append(self.query(q, c, **kwargs))

        return responses


def interactive_mode(model: GeneticLLM):
    """Run interactive Q&A session."""
    print("\n=== GeneticLLM Interactive Mode ===")
    print("Ask questions about genetics and genomics.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nThinking...")
        response = model.query(question)
        print(f"\nAnswer: {response}\n")
        print("-" * 60)


def batch_mode(
    model: GeneticLLM,
    input_path: Path,
    output_path: Path
):
    """Process a batch of questions from file."""
    with open(input_path) as f:
        data = json.load(f)

    # Extract questions
    questions = []
    for sample in data:
        messages = sample.get("messages", [])
        for msg in messages:
            if msg["role"] == "user":
                questions.append(msg["content"])
                break

    print(f"Processing {len(questions)} questions...")

    responses = model.batch_query(questions)

    with open(output_path, "w") as f:
        json.dump(responses, f, indent=2)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query GeneticLLM")
    parser.add_argument("--adapter", type=str, help="Path to LoRA adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--batch", type=Path, help="Input file for batch processing")
    parser.add_argument("--output", type=Path, help="Output file for batch predictions")
    parser.add_argument("--question", type=str, help="Single question to ask")

    args = parser.parse_args()

    # Initialize model
    model = GeneticLLM(
        base_model=args.base_model,
        adapter_path=args.adapter
    )

    if args.batch and args.output:
        batch_mode(model, args.batch, args.output)
    elif args.question:
        print(model.query(args.question))
    else:
        interactive_mode(model)
