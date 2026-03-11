# prcheck ✅

> Run this before every PR. It classifies your commits, catches common issues, and writes your PR description for you.

## Quick Setup

### 1. Install

```bash
cd prcheck
pip install -e .

# With AI commit classification (optional — uses HuggingFace model)
pip install -e ".[ai]"
```

### 2. Configure

Run the interactive setup wizard:

```bash
prcheck setup
```

This will prompt you for:
- **Anthropic API key** — for PR description generation ([get one here](https://console.anthropic.com/settings/keys))
- **HuggingFace token** — for private commit classifier models ([get one here](https://huggingface.co/settings/tokens))
- **Model ID** — the HuggingFace model for commit classification

Configuration is saved to `~/.prcheck/config.json`.

Alternatively, set environment variables or use a `.env` file:

```bash
cp .env.example .env
# Edit .env with your keys
```

### 3. Run

```bash
prcheck                        # Run all checks
prcheck --generate-pr-desc     # Also generate a PR description
prcheck --base develop         # Compare against a specific base branch
prcheck --json                 # Output results as JSON (for CI)
prcheck --model user/model     # Override the commit classifier model
```

## Example output

```
$ prcheck --generate-pr-desc

🔍 prcheck — comparing against 'main'
──────────────────────────────────────────────────
📝 Commits (4):

  [feat]       add JWT authentication to API (97%)
  [fix]        resolve null pointer in user service (94%)
  [refactor]   extract auth logic into separate module (91%)
  [chore]      bump dependency versions (88%)

📋 Checks:

  ✅ Diff size looks good: 183 lines (142+ / 41-)
  ⚠️  Found 2 TODO/FIXME comment(s) in new code
      # TODO: add refresh token support
      # FIXME: handle edge case for empty payload
  ✅ Test files present (2 test file(s) modified)
  ✅ No debug console logs added
  ✅ No obvious secrets detected

──────────────────────────────────────────────────
⚠️  1 warning(s) — review before opening PR

📄 Generating PR description...

## Summary
This PR adds JWT-based authentication to the API...

💾 Saved to PR_DESCRIPTION.md
```

## What it checks

| Check | Description |
|-------|-------------|
| 🏷️ Commit classification | Labels each commit as feat/fix/refactor/chore/docs/test/perf |
| 📏 Diff size | Warns at 400 lines, errors at 1000 lines |
| 📝 TODOs | Flags new TODO/FIXME/HACK/XXX comments |
| 🧪 Tests | Warns if source files changed but no test files |
| 🐛 Console logs | Detects debug `console.log` statements |
| 🔐 Secrets | Heuristic check for accidental API key commits |

## CI Integration

```yaml
# .github/workflows/pr-check.yml
- name: Run prcheck
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    PRCHECK_MODEL: your-username/commit-classifier
  run: |
    pip install prcheck
    prcheck --json > prcheck-results.json
```

## Powered by

- **[commit-classifier](../commit-classifier)** — fine-tuned DistilBERT for commit message classification
- **Claude API** — for PR description generation

## License

MIT
