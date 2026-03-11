# commit-classifier 🏷️

> Fine-tuned DistilBERT model that classifies git commit messages into conventional commit types.

**Labels**: `feat` · `fix` · `refactor` · `chore` · `docs` · `test` · `perf`

## Use the model (HuggingFace Hub)

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="your-username/commit-classifier")
classifier("add JWT authentication to API")
# [{'label': 'feat', 'score': 0.97}]
```

## Train it yourself

```bash
pip install -r requirements.txt
python train.py

# Push to HuggingFace Hub
python train.py --push-to-hub --hub-model-id your-username/commit-classifier
```

## Run inference

```bash
python train.py --predict "fix null pointer in user service"
# {"label": "fix", "confidence": 0.9821, "scores": {...}}
```

## Model details

| Property | Value |
|----------|-------|
| Base model | `distilbert-base-uncased` |
| Parameters | ~66M |
| Labels | 7 (feat, fix, refactor, chore, docs, test, perf) |
| Training size | ~56 samples (expand for production) |

## Used by `prcheck` CLI

This model powers the [`prcheck`](../prcheck) CLI tool to auto-tag commits in your PR.

## License

MIT
