# huggingface-mcp 🤗

> MCP server that gives Claude direct access to the HuggingFace Hub — search models, pull model cards, compare benchmarks, and explore trending datasets.

## Tools

| Tool | Description |
|------|-------------|
| `hf_search_models` | Search models by query, task, or library — sorted by downloads or likes |
| `hf_get_model_card` | Pull full metadata + card data for any model ID |
| `hf_search_datasets` | Search datasets by query or task category |
| `hf_compare_models` | Side-by-side comparison of 2–5 models |
| `hf_trending` | Get currently trending models or datasets |

## Example prompts

- *"Find the best small embedding model under 100M parameters"*
- *"Compare bert-base-uncased vs roberta-base vs distilbert"*
- *"What datasets are trending on HuggingFace right now?"*
- *"Pull the model card for mistralai/Mistral-7B-v0.1"*

## Quickstart

```bash
pip install huggingface-mcp
```

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "huggingface": {
      "command": "huggingface-mcp"
    }
  }
}
```

## Development

```bash
git clone https://github.com/yourusername/huggingface-mcp
cd huggingface-mcp
pip install -e ".[dev]"
python src/server.py
```

No API key required — uses the public HuggingFace API.

## License

MIT
