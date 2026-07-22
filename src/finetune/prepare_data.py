"""
Dataset preparation script for customer support fine-tuning.

Downloads and combines multiple datasets into a unified instruction format:
1. Bitext Customer Support - response generation
2. ServiceNow Incidents (synthetic) - classification + resolution
3. TriageIQ - sentiment/urgency/category

Usage:
    python prepare_data.py                          # Default: 10k samples
    python prepare_data.py --max-samples 5000       # Custom sample limit
    python prepare_data.py --upload-to-hub          # Push to Hugging Face
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


# ──── Formatting Functions ────


def format_bitext(example: Dict) -> Optional[Dict]:
    """Convert Bitext Customer Support to unified instruction format."""
    try:
        instruction = "Classify the following customer support inquiry and provide a helpful response."
        input_text = example.get("instruction", "") or example.get("input", "") or example.get("query", "")
        category = example.get("intent", "general")
        response = example.get("response", "") or example.get("output", "")

        if not input_text or not response:
            return None

        return {
            "instruction": instruction,
            "input": f"Ticket: {input_text}",
            "output": f"Category: {category}\nResponse: {response}",
            "source": "bitext",
        }
    except Exception:
        return None


def format_servicenow(example: Dict) -> Optional[Dict]:
    """Convert ServiceNow Incidents to unified instruction format."""
    try:
        instruction = "Classify this support ticket by category and urgency, then provide a resolution."
        description = example.get("short_description", "") or example.get("description", "")
        category = example.get("category", "Uncategorized")
        urgency = example.get("urgency", "3")
        resolution = example.get("resolution", "") or example.get("resolution_notes", "") or "Investigate and resolve."

        if not description:
            return None

        urgency_map = {"1": "Critical", "2": "High", "3": "Medium", "4": "Low", "5": "Low"}
        urgency_label = urgency_map.get(str(urgency).strip(), "Medium")

        return {
            "instruction": instruction,
            "input": f"Ticket: {description}",
            "output": f"Category: {category}\nUrgency: {urgency_label}\nResolution: {resolution}",
            "source": "servicenow",
        }
    except Exception:
        return None


def format_triageiq(example: Dict) -> Optional[Dict]:
    """Convert TriageIQ to unified instruction format."""
    try:
        instruction = "Analyze the sentiment, urgency, and category of this customer message."
        text = example.get("text", "") or example.get("message", "")
        sentiment = example.get("sentiment", "neutral")
        urgency = example.get("urgency", "medium")
        category = example.get("category", "general")

        if not text:
            return None

        return {
            "instruction": instruction,
            "input": f"Message: {text}",
            "output": f"Category: {category}\nUrgency: {urgency}\nSentiment: {sentiment}",
            "source": "triageiq",
        }
    except Exception:
        return None


# ──── Dataset Loading ────


def load_and_format_all(max_samples: int = 10000) -> pd.DataFrame:
    """Load all datasets and format into unified structure."""

    all_examples: List[Dict] = []
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    os.makedirs(data_dir, exist_ok=True)

    # ── 1. Bitext Customer Support ──
    print("[1/3] Loading Bitext Customer Support...")
    try:
        bitext = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
        for example in bitext:
            formatted = format_bitext(example)
            if formatted:
                all_examples.append(formatted)
        print(f"  ✓ {sum(1 for e in all_examples if e['source'] == 'bitext')} examples")
    except Exception as e:
        print(f"  ✗ Error loading Bitext: {e}")

    # ── 2. ServiceNow Incidents ──
    print("[2/3] Loading ServiceNow Incidents...")
    try:
        servicenow = load_dataset("6StringNinja/synthetic-servicenow-incidents", split="train")
        for example in servicenow:
            formatted = format_servicenow(example)
            if formatted:
                all_examples.append(formatted)
        print(f"  ✓ {sum(1 for e in all_examples if e['source'] == 'servicenow')} examples")
    except Exception as e:
        print(f"  ✗ Error loading ServiceNow: {e}")

    # ── 3. TriageIQ ──
    print("[3/3] Loading TriageIQ...")
    try:
        triageiq = load_dataset("coldstart88/triageiq-dataset", split="train")
        for example in triageiq:
            formatted = format_triageiq(example)
            if formatted:
                all_examples.append(formatted)
        print(f"  ✓ {sum(1 for e in all_examples if e['source'] == 'triageiq')} examples")
    except Exception as e:
        print(f"  ✗ Error loading TriageIQ: {e}")

    # ── Combine ──
    df = pd.DataFrame(all_examples)
    print(f"\nTotal examples: {len(df)}")
    print(f"Source breakdown: {df['source'].value_counts().to_dict()}")

    # Sample if too large
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=3407).reset_index(drop=True)
        print(f"Sampled to {max_samples} examples")

    return df


def split_and_save(df: pd.DataFrame, output_dir: str):
    """Split into train/eval and save to disk."""
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=3407)

    train_path = os.path.join(output_dir, "train.csv")
    eval_path = os.path.join(output_dir, "eval.csv")

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} examples)")
    print(f"  Eval:  {eval_path} ({len(eval_df)} examples)")


def upload_to_hub(train_dataset: Dataset, eval_dataset: Dataset, repo_name: str = "customer-support-unified"):
    """Push datasets to Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_name, exist_ok=True)

    train_dataset.push_to_hub(f"{repo_name}/train")
    eval_dataset.push_to_hub(f"{repo_name}/eval")
    print(f"Uploaded to Hugging Face Hub: {repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare customer support datasets for fine-tuning")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum number of samples to use")
    parser.add_argument("--upload-to-hub", action="store_true", help="Upload to Hugging Face Hub")
    parser.add_argument("--hub-repo", type=str, default="customer-support-unified", help="Hugging Face repo name")
    args = parser.parse_args()

    # Load and format data
    df = load_and_format_all(max_samples=args.max_samples)

    if df.empty:
        print("No data loaded. Exiting.")
        sys.exit(1)

    # Save to disk
    output_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    split_and_save(df, output_dir)

    # Optional: upload to Hugging Face Hub
    if args.upload_to_hub:
        train_dataset = Dataset.from_pandas(df.sample(frac=0.9, random_state=3407))
        eval_dataset = Dataset.from_pandas(df.drop(train_dataset.index))
        upload_to_hub(train_dataset, eval_dataset, args.hub_repo)

    print("\nDone! Ready for fine-tuning.")


if __name__ == "__main__":
    main()
