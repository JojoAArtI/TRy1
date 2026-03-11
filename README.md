# dev-trio 🛠️

Three interconnected projects for your ML Engineer resume — each publishable independently, together they tell a coherent story about **AI-powered developer tooling**.

```
dev-trio/
├── huggingface-mcp/      → MCP server: gives Claude access to HuggingFace Hub
├── commit-classifier/    → Fine-tuned DistilBERT on git commit messages (HF model)
└── prcheck/              → CLI tool: pre-PR checker powered by the classifier + Claude
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- [HuggingFace account](https://huggingface.co/join) — for model hosting
- [Anthropic API key](https://console.anthropic.com/settings/keys) — for PR description generation

### Step 1: Get your credentials

You'll need two keys:

| Credential | Where to get it | Used by |
|------------|----------------|---------|
| **HuggingFace Token** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | `commit-classifier`, `huggingface-mcp` |
| **Anthropic API Key** | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) | `prcheck` |

### Step 2: Set up each project

```bash
# 1. Train and publish the commit classifier
cd commit-classifier
pip install -e .
cp .env.example .env          # ← Add your HF_TOKEN and HF_USERNAME
python train.py               # Train the model
python train.py --push-to-hub --hub-model-id YOUR_USERNAME/commit-classifier

# 2. Set up the MCP server
cd ../huggingface-mcp
pip install -e .
cp .env.example .env          # ← Add your HF_TOKEN (optional)

# 3. Set up the PR checker
cd ../prcheck
pip install -e ".[ai]"
prcheck setup                 # ← Interactive setup for API keys
```

### Step 3: Use it!

```bash
# In any git repo with commits ahead of main:
prcheck --generate-pr-desc
```

---

## The story these projects tell

> *"I built an MCP server so Claude can search and compare HuggingFace models,
> fine-tuned a DistilBERT commit classifier and published it on HuggingFace Hub,
> then built a CLI tool that uses the model to audit PRs before they're opened."*

Each project generates a **metric** you can put on your resume:

| Project | Resume metric |
|---------|--------------|
| `huggingface-mcp` | Listed in MCP registry · X installs |
| `commit-classifier` | HuggingFace model · X downloads/month |
| `prcheck` | PyPI package · X installs · used in X GitHub Actions |

---

## Projects

### 1. `huggingface-mcp` — HuggingFace MCP Server
Gives Claude the ability to search models, pull model cards, compare benchmarks,
and explore trending datasets on the HuggingFace Hub — directly in conversation.

**Tools**: `hf_search_models`, `hf_get_model_card`, `hf_search_datasets`, `hf_compare_models`, `hf_trending`

```bash
pip install -e ./huggingface-mcp
```

→ [Full README](./huggingface-mcp/README.md)

---

### 2. `commit-classifier` — Fine-tuned Commit Classifier
DistilBERT fine-tuned to classify git commit messages into conventional commit types:
`feat` · `fix` · `refactor` · `chore` · `docs` · `test` · `perf`

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="YOUR_USERNAME/commit-classifier")
classifier("fix null pointer in payment service")
# [{'label': 'fix', 'score': 0.98}]
```

→ [Full README](./commit-classifier/README.md)

---

### 3. `prcheck` — Pre-PR Quality Checker CLI
Run before every PR. Classifies commits, detects TODOs/secrets/debug logs,
warns about large diffs, and generates a PR description using Claude.

```bash
pip install -e "./prcheck[ai]"
prcheck setup                  # Configure API keys
prcheck --generate-pr-desc     # Run checks + generate PR description
```

→ [Full README](./prcheck/README.md)

---

## Publishing checklist

- [ ] Push all three to separate GitHub repos
- [ ] Publish `huggingface-mcp` to PyPI → submit to MCP registry
- [ ] Train `commit-classifier` → push to HuggingFace Hub
- [ ] Publish `prcheck` to PyPI
- [ ] Add GitHub Actions badge + star count to each README
- [ ] Write a dev.to/Medium post: *"I built 3 AI dev tools in a weekend"*

---

## License

MIT
