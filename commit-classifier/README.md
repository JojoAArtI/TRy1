# commit-classifier 🏷️

> Fine-tuned DistilBERT model that classifies git commit messages into conventional commit types.

**Labels**: `feat` · `fix` · `refactor` · `chore` · `docs` · `test` · `perf`

## Quick Setup

### 1. Install

```bash
cd commit-classifier
pip install -e .
```

### 2. Configure HuggingFace token (for Hub publishing)

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
# Copy the example env file
cp .env.example .env
# Edit .env → set HF_TOKEN and HF_USERNAME
```

### 3. Train the model

```bash
# Train with the built-in seed dataset (105 samples)
python train.py

# Train with your own data (CSV or JSONL)
python train.py --data-file my_commits.csv

# Train AND publish to HuggingFace Hub
python train.py --push-to-hub --hub-model-id your-username/commit-classifier
```

### 4. Run inference

```bash
# From local model
python train.py --predict "fix null pointer in user service"
# {"label": "fix", "confidence": 0.9821, "scores": {...}}

# From a HuggingFace Hub model
python train.py --predict "add JWT auth" --model-path your-username/commit-classifier
```

## Use the model (Python)

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="your-username/commit-classifier")
classifier("add JWT authentication to API")
# [{'label': 'feat', 'score': 0.97}]
```

## Training data format

You can supply your own training data via `--data-file`:

**CSV format** (no header, or header with non-label values):
```csv
add user auth,feat
fix crash on login,fix
update README,docs
```

**JSONL format**:
```json
{"text": "add user auth", "label": "feat"}
{"text": "fix crash on login", "label": "fix"}
```

## Model details

| Property | Value |
|----------|-------|
| Base model | `distilbert-base-uncased` |
| Parameters | ~66M |
| Labels | 7 (feat, fix, refactor, chore, docs, test, perf) |
| Built-in training size | 105 samples (15 per label) |

## Used by `prcheck` CLI

This model powers the [`prcheck`](../prcheck) CLI tool to auto-tag commits in your PR.

## License

MIT
