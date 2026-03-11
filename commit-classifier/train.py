"""
commit-classifier: Fine-tune a DistilBERT model to classify git commit messages
into: feat, fix, refactor, chore, docs, test, perf

Upload to HuggingFace Hub after training.

Usage:
    python train.py
    python train.py --push-to-hub --hub-model-id your-username/commit-classifier
"""

import argparse
import json
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
]


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


def train(push_to_hub: bool = False, hub_model_id: str = None):
    MODEL_NAME = "distilbert-base-uncased"

    print("📦 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("📊 Building dataset...")
    dataset = build_dataset(SEED_DATA)
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
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train commit message classifier")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HuggingFace Hub after training")
    parser.add_argument("--hub-model-id", type=str, default=None, help="HF Hub model ID, e.g. 'username/commit-classifier'")
    parser.add_argument("--predict", type=str, default=None, help="Run inference on a single commit message")
    args = parser.parse_args()

    if args.predict:
        result = predict(args.predict)
        print(json.dumps(result, indent=2))
    else:
        train(push_to_hub=args.push_to_hub, hub_model_id=args.hub_model_id)
