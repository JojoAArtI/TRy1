# dev-trio 🛠️

Three interconnected projects for your ML Engineer resume — each publishable independently, together they tell a coherent story about **AI-powered developer tooling**.

```
dev-trio/
├── huggingface-mcp/      → MCP server: gives Claude access to HuggingFace Hub
├── commit-classifier/    → Fine-tuned DistilBERT on git commit messages (HF model)
└── prcheck/              → CLI tool: pre-PR checker powered by the classifier + Claude
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
pip install huggingface-mcp
```

→ [Full README](./huggingface-mcp/README.md)

---

### 2. `commit-classifier` — Fine-tuned Commit Classifier
DistilBERT fine-tuned to classify git commit messages into conventional commit types:
`feat` · `fix` · `refactor` · `chore` · `docs` · `test` · `perf`

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="your-username/commit-classifier")
classifier("fix null pointer in payment service")
# [{'label': 'fix', 'score': 0.98}]
```

→ [Full README](./commit-classifier/README.md)

---

### 3. `prcheck` — Pre-PR Quality Checker CLI
Run before every PR. Classifies commits, detects TODOs/secrets/debug logs,
warns about large diffs, and generates a PR description using Claude.

```bash
pip install prcheck
prcheck --generate-pr-desc
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
