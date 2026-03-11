# huggingface-mcp 🤗

> MCP server that gives Claude direct access to the HuggingFace Hub — search models, pull model cards, compare benchmarks, and explore trending datasets.

## Quick Setup

### 1. Install

```bash
cd huggingface-mcp
pip install -e .
```

### 2. Configure (optional but recommended)

Get a HuggingFace API token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
# Option A: Copy the example env file and fill in your token
cp .env.example .env
# Edit .env → set HF_TOKEN=hf_your_actual_token

# Option B: Set it directly in your shell
export HF_TOKEN=hf_your_actual_token
```

> **Note:** The server works without a token (public API), but a token gives you:
> - Access to gated/private models
> - Higher API rate limits
> - Access to private datasets

### 3. Run

```bash
# Direct
python src/server.py

# Or as a module
python -m src

# Or via the CLI entry point (after pip install)
huggingface-mcp
```

### 4. Add to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "huggingface": {
      "command": "huggingface-mcp",
      "env": {
        "HF_TOKEN": "hf_your_token_here"
      }
    }
  }
}
```

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

## Development

```bash
git clone https://github.com/yourusername/huggingface-mcp
cd huggingface-mcp
pip install -e ".[dev]"
python src/server.py
```

## License

MIT
