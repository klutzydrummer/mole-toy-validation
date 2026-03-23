#!/usr/bin/env python3
"""
Verification staleness checker for MoLE toy-validation.

This is the lightweight half of the verification pipeline. It tracks whether
model code has been verified against the reference library in references/components/.

The FULL verification (dispatching agents to cross-check code vs references)
is done by Claude Code locally on your machine. After agents complete and you
approve the results, call `update` to record the outcome. The cloud training
instance only ever calls `check` — no agents, no Claude, just a hash comparison.

Workflow:
  1. Change model code locally
  2. Ask Claude Code: "run full verification"  ← agents do the work here
  3. Review the report in references/verification/reports/
  4. python utils/verify.py update --result pass --report <path> [files...]
  5. git commit (including updated last_verified.json)
  6. Cloud: run_experiments.sh calls `python utils/verify.py check` — passes if hashes match

Commands:
  python utils/verify.py check [file ...]   Check staleness (exit 0=current, 1=stale)
  python utils/verify.py status             Print human-readable verification table
  python utils/verify.py update [file ...]  Record verification result (after agent run)
    --result pass|fail
    --report  path/to/report.md

If no files are given to `check` or `update`, all TRACKED_FILES are used.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

VERIFIED_JSON = REPO_ROOT / "references" / "verification" / "last_verified.json"

# Files whose content must be verified before training.
# Covers all architectural implementations — the math lives here.
TRACKED_FILES = [
    "phase1/model.py",
    "phase2/model.py",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def git_blob_hash(rel_path: str) -> str | None:
    """
    Return the git blob hash for a file — this is the SHA-1 of the file's
    content, independent of commit history. Changes only when the file changes.
    Returns None if git is unavailable or the file is untracked.
    """
    try:
        result = subprocess.run(
            ["git", "hash-object", rel_path],
            capture_output=True, text=True, cwd=REPO_ROOT, check=True,
        )
        h = result.stdout.strip()
        return h if h else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_record() -> dict:
    if VERIFIED_JSON.exists():
        with open(VERIFIED_JSON) as f:
            return json.load(f)
    return {"_note": "Updated by utils/verify.py update. Do not edit manually.", "files": {}}


def save_record(data: dict) -> None:
    VERIFIED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(VERIFIED_JSON, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_check(args) -> int:
    """
    Exit 0 if all tracked files are current (hash matches last_verified.json).
    Exit 1 if any are stale or have never been verified.
    Prints a summary either way.
    """
    files = args.files if args.files else TRACKED_FILES
    record = load_record()
    verified = record.get("files", {})

    stale = []
    for f in files:
        current = git_blob_hash(f)
        entry   = verified.get(f, {})
        if current is None:
            stale.append((f, "cannot compute hash (untracked or git unavailable)"))
        elif not entry:
            stale.append((f, "never verified"))
        elif entry.get("git_hash") != current:
            stale.append((f, f"modified since {entry.get('verified_at', 'unknown')}"))
        elif entry.get("result") != "pass":
            stale.append((f, f"last verification result was '{entry.get('result')}'"))

    if stale:
        print("=" * 60)
        print("  VERIFICATION STALE — training blocked")
        print("=" * 60)
        for f, reason in stale:
            print(f"  ✗  {f}")
            print(f"       {reason}")
        print()
        print("Run full verification locally in Claude Code:")
        print("  Tell Claude: 'run full verification'")
        print("  Then: python utils/verify.py update --result pass --report <path>")
        print()
        return 1
    else:
        print("Verification current — all tracked files match last_verified.json.")
        return 0


def cmd_status(args) -> int:
    """Print a human-readable table of verification status for all tracked files."""
    record   = load_record()
    verified = record.get("files", {})

    col_f  = 35
    col_s  = 14
    col_r  = 8
    col_t  = 22

    header = f"{'File':<{col_f}} {'Status':<{col_s}} {'Result':<{col_r}} {'Verified at':<{col_t}} Report"
    print(header)
    print("-" * (col_f + col_s + col_r + col_t + 30))

    for f in TRACKED_FILES:
        current = git_blob_hash(f)
        entry   = verified.get(f, {})

        if not entry:
            status = "never verified"
            result = "-"
            when   = "-"
            report = "-"
        elif current is None:
            status = "hash unavailable"
            result = entry.get("result", "-")
            when   = entry.get("verified_at", "-")
            report = entry.get("report", "-")
        elif entry.get("git_hash") != current:
            status = "STALE"
            result = entry.get("result", "-")
            when   = entry.get("verified_at", "-")
            report = entry.get("report", "-")
        elif entry.get("result") != "pass":
            status = "failed"
            result = entry.get("result", "-")
            when   = entry.get("verified_at", "-")
            report = entry.get("report", "-")
        else:
            status = "current"
            result = entry.get("result", "-")
            when   = entry.get("verified_at", "-")
            report = entry.get("report", "-")

        print(f"{f:<{col_f}} {status:<{col_s}} {result:<{col_r}} {when:<{col_t}} {report}")

    return 0


def cmd_update(args) -> int:
    """
    Record a verification result in last_verified.json.
    Called by Claude Code after agent verification is approved.
    """
    if args.result not in ("pass", "fail"):
        print(f"ERROR: --result must be 'pass' or 'fail', got '{args.result}'")
        return 1

    files = args.files if args.files else TRACKED_FILES
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data  = load_record()

    updated = []
    skipped = []
    for f in files:
        h = git_blob_hash(f)
        if h is None:
            print(f"WARNING: cannot compute hash for {f} — skipping")
            skipped.append(f)
            continue
        data.setdefault("files", {})[f] = {
            "git_hash":    h,
            "verified_at": now,
            "result":      args.result,
            "report":      args.report or "",
        }
        updated.append(f)

    save_record(data)

    if updated:
        print("Updated last_verified.json:")
        for f in updated:
            print(f"  {f}  result={args.result}  verified_at={now}")
    if skipped:
        print(f"Skipped (no hash): {', '.join(skipped)}")

    return 0 if not skipped else 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MoLE verification staleness checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # check
    p_check = sub.add_parser("check", help="Check if tracked files are current (exit 0) or stale (exit 1)")
    p_check.add_argument("files", nargs="*", help="Files to check (default: all TRACKED_FILES)")

    # status
    sub.add_parser("status", help="Print human-readable verification table")

    # update
    p_update = sub.add_parser("update", help="Record a verification result after agent run")
    p_update.add_argument("files", nargs="*", help="Files verified (default: all TRACKED_FILES)")
    p_update.add_argument("--result",  required=True, choices=["pass", "fail"])
    p_update.add_argument("--report",  default="",    help="Path to the verification report")

    args = parser.parse_args()

    if args.command == "check":
        sys.exit(cmd_check(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "update":
        sys.exit(cmd_update(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
