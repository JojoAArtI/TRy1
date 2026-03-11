"""
prcheck — pre-PR quality checker for developers

Runs before you open a pull request:
  • Classifies each commit using the commit-classifier HuggingFace model
  • Flags TODO/FIXME/HACK comments in changed files
  • Warns about large diffs
  • Detects missing tests
  • Auto-generates a PR description using Claude

Setup:
    prcheck setup              # Interactive setup — stores config in ~/.prcheck/config.json
    prcheck                    # Run all checks on current branch
    prcheck --generate-pr-desc # Also generate a PR description

Usage:
    prcheck                        # Run all checks on current branch
    prcheck --generate-pr-desc     # Also generate a PR description
    prcheck --base main            # Compare against a specific base branch
    prcheck --json                 # Output results as JSON
    prcheck setup                  # Configure API keys and model ID
"""

import subprocess
import sys
import re
import json
import argparse
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Load environment variables ────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".prcheck"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_MODEL_ID = "your-username/commit-classifier"


def load_config() -> dict:
    """Load configuration from ~/.prcheck/config.json and environment variables.

    Priority: env vars > config file > defaults
    """
    config = {
        "anthropic_api_key": None,
        "hf_token": None,
        "model_id": DEFAULT_MODEL_ID,
    }

    # Layer 1: config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                file_config = json.load(f)
            config.update({k: v for k, v in file_config.items() if v})
        except (json.JSONDecodeError, OSError):
            pass

    # Layer 2: environment variables (highest priority)
    env_mappings = {
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "HF_TOKEN": "hf_token",
        "PRCHECK_MODEL": "model_id",
    }
    for env_key, config_key in env_mappings.items():
        val = os.environ.get(env_key)
        if val:
            config[config_key] = val

    return config


def save_config(config: dict):
    """Save configuration to ~/.prcheck/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    # Protect the config file (contains secrets)
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass


def run_setup():
    """Interactive setup wizard for prcheck configuration."""
    print("🔧 prcheck setup")
    print("=" * 50)
    print()

    existing = load_config()

    # Anthropic API key
    print("1️⃣  Anthropic API Key (for PR description generation)")
    print("   Get one at: https://console.anthropic.com/settings/keys")
    current = existing.get("anthropic_api_key")
    if current:
        masked = current[:10] + "..." + current[-4:]
        print(f"   Current: {masked}")
    api_key = input("   Enter key (or press Enter to keep current): ").strip()
    if api_key:
        existing["anthropic_api_key"] = api_key
    print()

    # HuggingFace token
    print("2️⃣  HuggingFace Token (for private commit-classifier models)")
    print("   Get one at: https://huggingface.co/settings/tokens")
    current = existing.get("hf_token")
    if current:
        masked = current[:6] + "..." + current[-4:]
        print(f"   Current: {masked}")
    hf_token = input("   Enter token (or press Enter to skip/keep): ").strip()
    if hf_token:
        existing["hf_token"] = hf_token
    print()

    # Model ID
    print("3️⃣  Commit Classifier Model ID")
    print("   This is the HuggingFace model used to classify commits.")
    current = existing.get("model_id", DEFAULT_MODEL_ID)
    print(f"   Current: {current}")
    model_id = input("   Enter model ID (or press Enter to keep current): ").strip()
    if model_id:
        existing["model_id"] = model_id
    print()

    save_config(existing)
    print(f"✅ Configuration saved to {CONFIG_FILE}")
    print()

    # Validate
    config = load_config()
    if not config.get("anthropic_api_key"):
        print("⚠️  No Anthropic API key set — PR description generation won't work.")
        print("   You can still use all other checks.")
    else:
        print("✅ Anthropic API key configured")

    if config.get("model_id") == DEFAULT_MODEL_ID:
        print("⚠️  Using default model ID — update after publishing your model.")
    else:
        print(f"✅ Model: {config['model_id']}")
    print()


# ── Git helpers ───────────────────────────────────────────────────────────────

def git(*args: str) -> str:
    """Run a git command and return stdout. Raises on error."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def get_base_branch() -> str:
    """Auto-detect the most likely base branch."""
    for candidate in ("main", "master", "develop"):
        try:
            git("rev-parse", "--verify", candidate)
            return candidate
        except subprocess.CalledProcessError:
            continue
    return "main"


def get_commits(base: str) -> list[dict]:
    """Get commits between base and HEAD."""
    raw = git("log", f"{base}..HEAD", "--pretty=format:%H|%s|%an")
    if not raw:
        return []
    commits = []
    for line in raw.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            commits.append({"hash": parts[0][:8], "message": parts[1], "author": parts[2]})
    return commits


def get_diff(base: str) -> str:
    """Get full diff between base and HEAD."""
    return git("diff", f"{base}..HEAD")


def get_changed_files(base: str) -> list[str]:
    """Get list of changed file paths."""
    raw = git("diff", "--name-only", f"{base}..HEAD")
    return [f for f in raw.splitlines() if f]


def count_diff_lines(diff: str) -> dict:
    added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return {"added": added, "removed": removed, "total": added + removed}


# ── Checks ────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)
    severity: str = "warning"  # "info", "warning", "error"


def check_diff_size(diff: str, warn_threshold=400, error_threshold=1000) -> CheckResult:
    counts = count_diff_lines(diff)
    total = counts["total"]
    if total >= error_threshold:
        return CheckResult(
            name="diff_size", passed=False, severity="error",
            message=f"🚨 Very large diff: {total} lines changed ({counts['added']}+ / {counts['removed']}-)",
            details=["Consider splitting this PR into smaller, focused changes."]
        )
    if total >= warn_threshold:
        return CheckResult(
            name="diff_size", passed=False, severity="warning",
            message=f"⚠️  Large diff: {total} lines changed ({counts['added']}+ / {counts['removed']}-)",
            details=["PRs under 400 lines are reviewed faster and more thoroughly."]
        )
    return CheckResult(
        name="diff_size", passed=True, severity="info",
        message=f"✅ Diff size looks good: {total} lines ({counts['added']}+ / {counts['removed']}-)"
    )


def check_todos(diff: str) -> CheckResult:
    pattern = re.compile(r'^\+.*\b(TODO|FIXME|HACK|XXX|BUG)\b', re.MULTILINE)
    matches = pattern.findall(diff)
    lines = [l.lstrip("+").strip() for l in diff.splitlines()
             if l.startswith("+") and re.search(r'\b(TODO|FIXME|HACK|XXX|BUG)\b', l)]
    if lines:
        return CheckResult(
            name="todos", passed=False, severity="warning",
            message=f"⚠️  Found {len(lines)} TODO/FIXME/HACK comment(s) in new code",
            details=lines[:5]
        )
    return CheckResult(name="todos", passed=True, severity="info", message="✅ No TODO/FIXME/HACK comments added")


def check_tests(changed_files: list[str], diff: str) -> CheckResult:
    src_files = [f for f in changed_files if f.endswith((".py", ".ts", ".js")) and "test" not in f and "spec" not in f]
    test_files = [f for f in changed_files if "test" in f.lower() or "spec" in f.lower()]

    if src_files and not test_files:
        return CheckResult(
            name="tests", passed=False, severity="warning",
            message=f"⚠️  No test files changed (but {len(src_files)} source file(s) were modified)",
            details=[f"  - {f}" for f in src_files[:5]]
        )
    return CheckResult(
        name="tests", passed=True, severity="info",
        message=f"✅ Test files present ({len(test_files)} test file(s) modified)"
    )


def check_console_logs(diff: str) -> CheckResult:
    pattern = re.compile(r'^\+.*\bconsole\.(log|debug|warn)\b', re.MULTILINE)
    matches = pattern.findall(diff)
    if matches:
        return CheckResult(
            name="console_logs", passed=False, severity="warning",
            message=f"⚠️  Found {len(matches)} console.log/debug statement(s) in new code",
            details=["Remove debug logs before merging to production."]
        )
    return CheckResult(name="console_logs", passed=True, severity="info", message="✅ No debug console logs added")


def check_secrets(diff: str) -> CheckResult:
    """Basic heuristic check for accidental secret commits."""
    patterns = [
        (r'(?i)(api_key|secret_key|password|token)\s*=\s*["\'][^"\']{8,}["\']', "hardcoded secret"),
        (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API key pattern"),
        (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token"),
        (r'sk-ant-api03-[a-zA-Z0-9]{32,}', "Anthropic API key pattern"),
        (r'hf_[a-zA-Z0-9]{34,}', "HuggingFace token pattern"),
    ]
    found = []
    for line in diff.splitlines():
        if not line.startswith("+"):
            continue
        for pat, label in patterns:
            if re.search(pat, line):
                found.append(f"{label}: {line[:80].lstrip('+').strip()}")
    if found:
        return CheckResult(
            name="secrets", passed=False, severity="error",
            message=f"🚨 Possible secret/credential detected in diff!",
            details=found[:3]
        )
    return CheckResult(name="secrets", passed=True, severity="info", message="✅ No obvious secrets detected")


def classify_commits(commits: list[dict], config: dict) -> list[dict]:
    """Classify commits using the HuggingFace model if available, else use regex heuristics."""
    model_id = config.get("model_id", DEFAULT_MODEL_ID)

    try:
        from transformers import pipeline
        print(f"   Using model: {model_id}")
        classifier = pipeline("text-classification", model=model_id, device=-1)
        for c in commits:
            result = classifier(c["message"])[0]
            c["label"] = result["label"]
            c["confidence"] = round(result["score"], 3)
    except ImportError:
        print("   ⚠️  transformers not installed — using regex fallback")
        print("   💡 Install with: pip install 'prcheck[ai]'")
        _classify_regex_fallback(commits)
    except Exception as e:
        print(f"   ⚠️  Model loading failed ({e}) — using regex fallback")
        _classify_regex_fallback(commits)

    return commits


def _classify_regex_fallback(commits: list[dict]):
    """Fallback: simple prefix heuristic for commit classification."""
    prefix_map = {
        "feat": r'^(feat|add|new|implement|support|create)',
        "fix": r'^(fix|bug|hotfix|patch|resolve|correct)',
        "refactor": r'^(refactor|clean|rename|move|extract|split|simplify|rewrite|convert|decouple)',
        "docs": r'^(docs|doc|readme|comment|changelog|document)',
        "test": r'^(test|spec|coverage)',
        "perf": r'^(perf|optim|cache|speed|lazy|reduce|compress|batch|memoize)',
        "chore": r'^(chore|bump|upgrade|update|ci|cd|build|deps|configure|migrate|remove|clean|sync|pin|archive)',
    }
    for c in commits:
        msg = c["message"].lower()
        label = "chore"
        for lbl, pat in prefix_map.items():
            if re.search(pat, msg):
                label = lbl
                break
        c["label"] = label
        c["confidence"] = None


# ── PR Description Generator ─────────────────────────────────────────────────

def generate_pr_description(commits: list[dict], diff_summary: dict,
                            changed_files: list[str], config: dict) -> str:
    """Call Claude API to generate a PR description."""
    api_key = config.get("anthropic_api_key")
    if not api_key:
        return (
            "(Could not generate PR description: No Anthropic API key configured.\n"
            " Run 'prcheck setup' or set ANTHROPIC_API_KEY environment variable.\n"
            " Get a key at: https://console.anthropic.com/settings/keys)"
        )

    try:
        import urllib.request

        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 600,
            "messages": [{
                "role": "user",
                "content": f"""Generate a concise GitHub PR description based on:

COMMITS:
{json.dumps([{"message": c["message"], "type": c.get("label", "?")} for c in commits], indent=2)}

CHANGED FILES:
{json.dumps(changed_files[:20], indent=2)}

DIFF STATS:
{json.dumps(diff_summary, indent=2)}

Format:
## Summary
(2-3 sentence summary of what this PR does)

## Changes
(bullet list of key changes grouped by type)

## Testing
(what was tested or how to test)
"""
            }]
        }

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except Exception as e:
        return f"(Could not generate PR description: {e})"


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_checks(base: str, generate_desc: bool, output_json: bool, config: dict):
    print(f"\n🔍 prcheck — comparing against '{base}'\n{'─'*50}")

    try:
        commits = get_commits(base)
        diff = get_diff(base)
        changed_files = get_changed_files(base)
    except subprocess.CalledProcessError as e:
        print(f"❌ Git error: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    if not commits and not diff:
        print("ℹ️  No changes found between HEAD and base branch.")
        sys.exit(0)

    # Classify commits
    commits = classify_commits(commits, config)

    # Run checks
    checks = [
        check_diff_size(diff),
        check_todos(diff),
        check_tests(changed_files, diff),
        check_console_logs(diff),
        check_secrets(diff),
    ]

    diff_stats = count_diff_lines(diff)

    if output_json:
        result = {
            "base": base,
            "commits": commits,
            "diff_stats": diff_stats,
            "changed_files": changed_files,
            "checks": [asdict(c) for c in checks],
        }
        print(json.dumps(result, indent=2))
        return

    # Print commits
    print(f"📝 Commits ({len(commits)}):\n")
    for c in commits:
        badge = f"[{c.get('label', '?')}]"
        conf = f" ({int(c['confidence']*100)}%)" if c.get("confidence") else ""
        print(f"  {badge:<12} {c['message']}{conf}")

    # Print check results
    print(f"\n📋 Checks:\n")
    all_passed = True
    for check in checks:
        print(f"  {check.message}")
        if check.details:
            for d in check.details:
                print(f"      {d}")
        if not check.passed and check.severity == "error":
            all_passed = False

    # Summary
    errors = [c for c in checks if not c.passed and c.severity == "error"]
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]
    print(f"\n{'─'*50}")
    if errors:
        print(f"🚨 {len(errors)} error(s), {len(warnings)} warning(s) — fix errors before opening PR")
    elif warnings:
        print(f"⚠️  {len(warnings)} warning(s) — review before opening PR")
    else:
        print("✅ All checks passed — you're good to open the PR!")

    # Generate PR description
    if generate_desc:
        print(f"\n{'─'*50}\n📄 Generating PR description...\n")
        desc = generate_pr_description(commits, diff_stats, changed_files, config)
        print(desc)

        # Write to file for easy copy-paste
        Path("PR_DESCRIPTION.md").write_text(desc)
        print("\n💾 Saved to PR_DESCRIPTION.md")


def main():
    parser = argparse.ArgumentParser(
        description="prcheck — pre-PR quality checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  prcheck                        Run all checks on current branch
  prcheck setup                  Configure API keys and preferences

Examples:
  prcheck                        Run checks (compare against auto-detected base)
  prcheck --base develop         Compare against 'develop' branch
  prcheck --generate-pr-desc     Generate a PR description with Claude
  prcheck --json                 Output results as JSON (for CI)
  prcheck --model user/model     Override the commit classifier model
        """
    )

    # Handle 'setup' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        run_setup()
        return

    parser.add_argument("--base", "-b", default=None, help="Base branch to compare against (default: auto-detect)")
    parser.add_argument("--generate-pr-desc", "-g", action="store_true", help="Generate a PR description using Claude")
    parser.add_argument("--json", action="store_true", dest="output_json", help="Output results as JSON")
    parser.add_argument("--model", type=str, default=None, help="Override the commit classifier model ID")
    args = parser.parse_args()

    config = load_config()

    # CLI flag overrides config
    if args.model:
        config["model_id"] = args.model

    base = args.base or get_base_branch()
    run_checks(base=base, generate_desc=args.generate_pr_desc, output_json=args.output_json, config=config)


if __name__ == "__main__":
    main()
