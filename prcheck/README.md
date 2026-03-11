# prcheck ✅

> Run this before every PR. It classifies your commits, catches common issues, and writes your PR description for you.

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
This PR adds JWT-based authentication to the API, fixes a null pointer
exception in the user service, and refactors auth logic into a dedicated
module for better separation of concerns.
...

💾 Saved to PR_DESCRIPTION.md
```

## Install

```bash
pip install prcheck

# With AI commit classification (uses your-username/commit-classifier from HuggingFace)
pip install "prcheck[ai]"
```

## Usage

```bash
prcheck                        # Run all checks
prcheck --generate-pr-desc     # Also generate a PR description
prcheck --base develop         # Compare against a specific base branch
prcheck --json                 # Output results as JSON (for CI)
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
  run: |
    pip install prcheck
    prcheck --json > prcheck-results.json
```

## Powered by

- **[commit-classifier](https://huggingface.co/your-username/commit-classifier)** — fine-tuned DistilBERT for commit message classification
- **Claude API** — for PR description generation

## License

MIT
