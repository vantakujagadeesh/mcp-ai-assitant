"""
Training script for QLoRA fine-tuning of Llama 3.1 8B on customer support data.

This is the local-equivalent of the Colab notebook. For Colab, use the notebook instead.

Usage:
    python train.py                          # Train with defaults
    python train.py --model Qwen2.5-7B       # Switch base model
    python train.py --max-steps 1000         # Longer training
"""

import argparse
import logging
import os
import sys

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from prepare_data import load_and_format_all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupportFineTuner:
    """Handles QLoRA fine-tuning for customer support models."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_base_model(self):
        """Load base model with 4-bit quantization and LoRA."""
        model_name = self.config.get(
            "base_model",
            "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
        )
        max_seq_length = self.config.get("max_seq_length", 2048)

        logger.info(f"Loading base model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
        )

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.get("lora_r", 16),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
            loftq_config=None,
        )

        self.model = model
        self.tokenizer = tokenizer
        logger.info(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model, tokenizer

    def tokenize_dataset(self, dataset: Dataset, max_seq_length: int = 2048) -> Dataset:
        """Apply chat template and tokenize."""
        def tokenize_function(examples):
            texts = []
            for i in range(len(examples["instruction"])):
                messages = [
                    {"role": "system", "content": examples["instruction"][i]},
                    {"role": "user", "content": examples["input"][i]},
                    {"role": "assistant", "content": examples["output"][i]},
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                texts.append(text)

            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            batch_size=32,
        )

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Run training."""
        output_dir = self.config.get("output_dir", "./results/llama-support")
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.get("batch_size", 2),
            gradient_accumulation_steps=self.config.get("gradient_accumulation", 4),
            warmup_steps=self.config.get("warmup_steps", 5),
            max_steps=self.config.get("max_steps", 500),
            learning_rate=self.config.get("learning_rate", 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            report_to="none",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            max_seq_length=self.config.get("max_seq_length", 2048),
            dataset_text_field="text",
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")

        # Save
        save_path = self.config.get("save_path", "./llama-support-lora")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

        return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 for customer support")
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="./results/llama-support")
    parser.add_argument("--save-path", type=str, default="./llama-support-lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    config = {
        "base_model": args.model,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_samples": args.max_samples,
        "output_dir": args.output_dir,
        "save_path": args.save_path,
        "max_seq_length": args.max_seq_length,
        "lora_r": 16,
        "lora_alpha": 32,
    }

    # Prepare data
    print("Preparing dataset...")
    df = load_and_format_all(max_samples=args.max_samples)
    if df.empty:
        print("No data loaded. Exiting.")
        sys.exit(1)

    train_df = df.sample(frac=0.9, random_state=3407)
    eval_df = df.drop(train_df.index)
    print(f"Train: {len(train_df)}, Eval: {len(eval_df)}")

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    # Fine-tune
    tuner = SupportFineTuner(config)
    tuner.load_base_model()

    print("Tokenizing datasets...")
    tokenized_train = tuner.tokenize_dataset(train_dataset)
    tokenized_eval = tuner.tokenize_dataset(eval_dataset)

    tuner.train(tokenized_train, tokenized_eval)

    print("\n✓ Training complete!")
    print(f"  Model saved to: {args.save_path}")
    print("  Next: Run evaluation with: python src/eval/benchmark.py")


if __name__ == "__main__":
    main()
