"""
commit-classifier: Fine-tune a DistilBERT model to classify git commit messages
into: feat, fix, refactor, chore, docs, test, perf

Upload to HuggingFace Hub after training.

Setup:
    1. pip install -e .
    2. Copy .env.example to .env and set your HF_TOKEN
    3. python train.py                                    # Train locally
    4. python train.py --push-to-hub --hub-model-id user/commit-classifier  # Train & publish

Usage:
    python train.py
    python train.py --push-to-hub --hub-model-id your-username/commit-classifier
    python train.py --predict "fix null pointer in user service"
    python train.py --data-file data.csv     # Load training data from a CSV file
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
import evaluate

# ── Load environment variables ────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")

# ── Label definitions ────────────────────────────────────────────────────────

LABELS = ["feat", "fix", "refactor", "chore", "docs", "test", "perf"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# ── Seed dataset (extend this or load from a CSV/JSONL file) ─────────────────

SEED_DATA = [
    # feat
    ("add user authentication with JWT", "feat"),
    ("implement dark mode toggle", "feat"),
    ("add support for markdown rendering", "feat"),
    ("new endpoint for bulk user creation", "feat"),
    ("implement rate limiting middleware", "feat"),
    ("add CSV export functionality", "feat"),
    ("support multi-language in dashboard", "feat"),
    ("add websocket support for real-time updates", "feat"),
    ("implement oauth2 login with google", "feat"),
    ("create admin panel for user management", "feat"),
    ("add file upload support", "feat"),
    ("implement search functionality with elasticsearch", "feat"),
    ("add two-factor authentication", "feat"),
    ("create user notification system", "feat"),
    ("add api versioning support", "feat"),
    # fix
    ("fix null pointer exception in user service", "fix"),
    ("resolve broken redirect after login", "fix"),
    ("patch XSS vulnerability in comment field", "fix"),
    ("fix race condition in async handler", "fix"),
    ("correct wrong status code on 404", "fix"),
    ("fix memory leak in WebSocket handler", "fix"),
    ("resolve incorrect timezone conversion", "fix"),
    ("fix pagination off-by-one error", "fix"),
    ("hotfix: crash on empty input in search", "fix"),
    ("fix broken tests after schema migration", "fix"),
    ("fix CORS issue with API gateway", "fix"),
    ("resolve deadlock in database transactions", "fix"),
    ("fix session expiration handling", "fix"),
    ("correct CSV parsing for quoted fields", "fix"),
    ("fix email validation regex false positives", "fix"),
    # refactor
    ("extract auth logic into separate module", "refactor"),
    ("simplify user model with dataclasses", "refactor"),
    ("rename misleading variable names in parser", "refactor"),
    ("move database calls to repository layer", "refactor"),
    ("refactor config loading to use env vars", "refactor"),
    ("clean up unused imports across codebase", "refactor"),
    ("replace for loops with list comprehensions", "refactor"),
    ("split monolithic controller into services", "refactor"),
    ("consolidate duplicate validation logic", "refactor"),
    ("rewrite legacy auth module with modern patterns", "refactor"),
    ("convert callbacks to async/await pattern", "refactor"),
    ("extract shared utilities into a common module", "refactor"),
    ("reorganize project directory structure", "refactor"),
    ("simplify error handling with custom exceptions", "refactor"),
    ("decouple business logic from framework code", "refactor"),
    # chore
    ("bump dependency versions", "chore"),
    ("update .gitignore for node_modules", "chore"),
    ("configure pre-commit hooks", "chore"),
    ("add docker-compose for local dev", "chore"),
    ("set up CI/CD pipeline on GitHub Actions", "chore"),
    ("migrate from pip to poetry", "chore"),
    ("remove deprecated API usage warnings", "chore"),
    ("clean up old migration files", "chore"),
    ("upgrade node to v20", "chore"),
    ("sync lockfile after dependency update", "chore"),
    ("update base docker image to alpine 3.19", "chore"),
    ("configure eslint with stricter rules", "chore"),
    ("add editorconfig for consistent formatting", "chore"),
    ("pin all dependency versions in lockfile", "chore"),
    ("archive unused feature branches", "chore"),
    # docs
    ("add README for authentication module", "docs"),
    ("document public API endpoints with OpenAPI", "docs"),
    ("update CHANGELOG for v2.0 release", "docs"),
    ("add inline comments to complex algorithm", "docs"),
    ("write contributing guide for OSS release", "docs"),
    ("update deployment instructions in wiki", "docs"),
    ("add docstrings to all public functions", "docs"),
    ("fix broken links in documentation", "docs"),
    ("clarify setup instructions in README", "docs"),
    ("document environment variables required", "docs"),
    ("add architecture decision records (ADRs)", "docs"),
    ("create troubleshooting FAQ section", "docs"),
    ("document database schema with diagrams", "docs"),
    ("add code examples to API reference", "docs"),
    ("write migration guide from v1 to v2", "docs"),
    # test
    ("add unit tests for payment processor", "test"),
    ("write integration tests for auth flow", "test"),
    ("add edge case tests for null inputs", "test"),
    ("increase test coverage to 80%", "test"),
    ("add snapshot tests for dashboard components", "test"),
    ("write end-to-end tests with playwright", "test"),
    ("mock external API calls in unit tests", "test"),
    ("add regression test for issue #412", "test"),
    ("parameterize tests for multiple locales", "test"),
    ("add load test for checkout endpoint", "test"),
    ("add contract tests for microservice APIs", "test"),
    ("test error handling for network timeouts", "test"),
    ("add fuzz testing for input validation", "test"),
    ("write tests for database migration rollback", "test"),
    ("add benchmark tests for critical paths", "test"),
    # perf
    ("cache database queries with redis", "perf"),
    ("lazy load images on product page", "perf"),
    ("optimize N+1 queries in user listing", "perf"),
    ("add database index on email column", "perf"),
    ("reduce bundle size with tree shaking", "perf"),
    ("compress API responses with gzip", "perf"),
    ("batch API calls to reduce round trips", "perf"),
    ("memoize expensive computations in dashboard", "perf"),
    ("parallelize data ingestion pipeline", "perf"),
    ("optimize image compression on upload", "perf"),
    ("implement connection pooling for database", "perf"),
    ("add CDN caching for static assets", "perf"),
    ("reduce memory footprint of in-memory cache", "perf"),
    ("optimize SQL queries with query plan analysis", "perf"),
    ("use streaming responses for large file downloads", "perf"),
]


def load_data_from_file(filepath: str) -> list[tuple[str, str]]:
    """Load training data from a CSV or JSONL file.

    CSV format: two columns — text, label (with or without header)
    JSONL format: {"text": "...", "label": "..."}

    Args:
        filepath: Path to the data file

    Returns:
        List of (text, label) tuples
    """
    path = Path(filepath)
    data = []

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                obj = json.loads(line.strip())
                text = obj.get("text") or obj.get("message") or obj.get("commit")
                label = obj.get("label") or obj.get("type")
                if text and label and label in LABELS:
                    data.append((text, label))
    elif path.suffix in (".csv", ".tsv"):
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if len(row) >= 2:
                    text, label = row[0].strip(), row[1].strip()
                    # Skip header row
                    if label in LABELS:
                        data.append((text, label))
    else:
        print(f"❌ Unsupported file format: {path.suffix}. Use .csv, .tsv, or .jsonl")
        sys.exit(1)

    return data


def build_dataset(data: list[tuple[str, str]]) -> DatasetDict:
    """Convert seed data into a HuggingFace DatasetDict with train/test split."""
    texts = [d[0] for d in data]
    labels = [LABEL2ID[d[1]] for d in data]
    ds = Dataset.from_dict({"text": texts, "label": labels})
    return ds.train_test_split(test_size=0.2, seed=42)


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=128)


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def setup_hf_auth(token: str = None):
    """Authenticate with HuggingFace Hub for model pushing.

    Args:
        token: HF API token. If None, tries HF_TOKEN env var.
    """
    token = token or HF_TOKEN
    if not token:
        print("❌ HuggingFace token required to push to Hub.")
        print("   Set it via:")
        print("   • --hf-token flag")
        print("   • HF_TOKEN environment variable")
        print("   • .env file (copy .env.example to .env)")
        print()
        print("   Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    from huggingface_hub import login
    login(token=token)
    print("✅ Authenticated with HuggingFace Hub")


def train(push_to_hub: bool = False, hub_model_id: str = None,
          hf_token: str = None, data_file: str = None):
    MODEL_NAME = "distilbert-base-uncased"

    # Authenticate if pushing to Hub
    if push_to_hub:
        setup_hf_auth(hf_token)
        if not hub_model_id:
            print("❌ --hub-model-id required when pushing to Hub.")
            print(f"   Example: --hub-model-id {HF_USERNAME}/commit-classifier")
            sys.exit(1)

    # Load training data
    if data_file:
        print(f"📂 Loading training data from {data_file}...")
        training_data = load_data_from_file(data_file)
        print(f"   Loaded {len(training_data)} samples")
        if not training_data:
            print("❌ No valid training samples found in file.")
            sys.exit(1)
    else:
        training_data = SEED_DATA
        print(f"📦 Using built-in seed dataset ({len(training_data)} samples)")

    print("📦 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("📊 Building dataset...")
    dataset = build_dataset(training_data)
    tokenized = dataset.map(lambda b: tokenize(b, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./commit-classifier-model",
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hf_token or HF_TOKEN,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("🚀 Training...")
    trainer.train()

    print("✅ Saving model locally to ./commit-classifier-model")
    trainer.save_model()
    tokenizer.save_pretrained("./commit-classifier-model")

    # Save label map for inference
    with open("./commit-classifier-model/label_map.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    if push_to_hub:
        print(f"📤 Pushing to HuggingFace Hub as {hub_model_id}...")
        trainer.push_to_hub()
        print(f"✅ Published at https://huggingface.co/{hub_model_id}")


def predict(text: str, model_path: str = "./commit-classifier-model") -> dict:
    """Run inference on a commit message."""
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found at '{model_path}'.")
        print("   Train the model first with: python train.py")
        print("   Or specify a HuggingFace model: --model-path username/commit-classifier")
        sys.exit(1)

    import torch
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    return {
        "label": ID2LABEL[probs.argmax().item()],
        "confidence": round(probs.max().item(), 4),
        "scores": {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probs)},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and use the commit message classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                                           # Train with seed data
  python train.py --data-file commits.csv                   # Train with custom data
  python train.py --predict "fix null pointer in user svc"  # Run inference
  python train.py --push-to-hub --hub-model-id user/model   # Train & publish
        """
    )
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push model to HuggingFace Hub after training")
    parser.add_argument("--hub-model-id", type=str, default=None,
                        help="HF Hub model ID, e.g. 'username/commit-classifier'")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to CSV/JSONL file with training data (text, label columns)")
    parser.add_argument("--predict", type=str, default=None,
                        help="Run inference on a single commit message")
    parser.add_argument("--model-path", type=str, default="./commit-classifier-model",
                        help="Path to trained model (default: ./commit-classifier-model)")
    args = parser.parse_args()

    if args.predict:
        result = predict(args.predict, model_path=args.model_path)
        print(json.dumps(result, indent=2))
    else:
        train(
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            hf_token=args.hf_token,
            data_file=args.data_file,
        )


if __name__ == "__main__":
    main()
