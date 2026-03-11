"""
prcheck — pre-PR quality checker for developers

Runs before you open a pull request:
  • Classifies each commit using the commit-classifier HuggingFace model
  • Flags TODO/FIXME/HACK comments in changed files
  • Warns about large diffs
  • Detects missing tests
  • Auto-generates a PR description using Claude

Usage:
    prcheck                        # Run all checks on current branch
    prcheck --generate-pr-desc     # Also generate a PR description
    prcheck --base main            # Compare against a specific base branch
    prcheck --json                 # Output results as JSON
"""

import subprocess
import sys
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

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


def classify_commits(commits: list[dict]) -> list[dict]:
    """Classify commits using the HuggingFace model if available, else use regex heuristics."""
    try:
        from transformers import pipeline
        classifier = pipeline("text-classification", model="your-username/commit-classifier", device=-1)
        for c in commits:
            result = classifier(c["message"])[0]
            c["label"] = result["label"]
            c["confidence"] = round(result["score"], 3)
    except Exception:
        # Fallback: simple prefix heuristic
        prefix_map = {
            "feat": r'^(feat|add|new|implement|support)',
            "fix": r'^(fix|bug|hotfix|patch|resolve|correct)',
            "refactor": r'^(refactor|clean|rename|move|extract|split|simplify)',
            "docs": r'^(docs|doc|readme|comment|changelog)',
            "test": r'^(test|spec|coverage)',
            "perf": r'^(perf|optim|cache|speed|lazy|reduce)',
            "chore": r'^(chore|bump|upgrade|update|ci|cd|build|deps)',
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
    return commits


# ── PR Description Generator ─────────────────────────────────────────────────

def generate_pr_description(commits: list[dict], diff_summary: dict, changed_files: list[str]) -> str:
    """Call Claude API to generate a PR description."""
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
                "anthropic-version": "2023-06-01",
            }
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except Exception as e:
        return f"(Could not generate PR description: {e})"


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_checks(base: str, generate_desc: bool, output_json: bool):
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
    commits = classify_commits(commits)

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
        desc = generate_pr_description(commits, diff_stats, changed_files)
        print(desc)

        # Write to file for easy copy-paste
        Path("PR_DESCRIPTION.md").write_text(desc)
        print("\n💾 Saved to PR_DESCRIPTION.md")


def main():
    parser = argparse.ArgumentParser(
        description="prcheck — pre-PR quality checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--base", "-b", default=None, help="Base branch to compare against (default: auto-detect)")
    parser.add_argument("--generate-pr-desc", "-g", action="store_true", help="Generate a PR description using Claude")
    parser.add_argument("--json", action="store_true", dest="output_json", help="Output results as JSON")
    args = parser.parse_args()

    base = args.base or get_base_branch()
    run_checks(base=base, generate_desc=args.generate_pr_desc, output_json=args.output_json)


if __name__ == "__main__":
    main()
