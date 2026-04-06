#!/usr/bin/env python3
"""
Verification staleness checker for MoLE toy-validation.

Schema v2: tracks components (named groups of files) instead of individual files.
Each component corresponds to a spec in references/components/ and maps to one or
more implementation files that are jointly hashed.

Workflow:
  1. Change a component file locally
  2. Ask Claude Code: "run full verification for <component>"  ← agents do the work
  3. Review the report in references/verification/reports/
  4. python utils/verify.py update --result pass --report <path> <component> [...]
  5. git commit (including updated last_verified.json)
  6. Cloud: run_experiments.sh calls `python utils/verify.py check` — passes if all current

Commands:
  python utils/verify.py check [component ...]   Check staleness (exit 0=current, 1=stale)
  python utils/verify.py status                  Print human-readable verification table
  python utils/verify.py update <component> ...  Record verification result (after agent run)
    --result pass|fail
    --report  path/to/report.md
  python utils/verify.py migrate                 Migrate v1 schema to v2 (first-time only)

If no components are given to `check`, all TRACKED_COMPONENTS are checked.
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

# Directories that contain verifiable component files.
COMPONENT_DIRS = [
    "phase1/components",
    "phase2/components",
]

# Files in COMPONENT_DIRS that are intentionally NOT tracked by any component.
# Every file added here must have a documented reason.
UNTRACKED_EXEMPT = {
    # __init__.py files: empty namespace packages, no implementation.
    "phase1/components/__init__.py",
    "phase2/components/__init__.py",
    # transformer_block.py: wiring only — combines components but contains no
    # mathematical implementation. No spec in references/components/. See
    # phase1/components/CLAUDE.md: "wiring only — no component spec".
    "phase1/components/transformer_block.py",
}

# Components: named groups of files tracked together.
# A component is stale when ANY of its files changes since last verification.
# Multiple components may share a file (e.g., _shared.py, zone_e.py).
TRACKED_COMPONENTS: dict[str, list[str]] = {
    "attention_rope_norms": [
        "phase1/components/attention_rope_norms.py",
        "phase1/components/_shared.py",
    ],
    "mla_attention": [
        "phase1/components/mla_attention.py",
        "phase1/components/attention_rope_norms.py",
        "phase1/components/_shared.py",
    ],
    "diff_attention": [
        "phase1/components/diff_attention.py",
        "phase1/components/attention_rope_norms.py",
        "phase1/components/_shared.py",
    ],
    "mhc": [
        "phase1/components/mhc.py",
    ],
    "mol_ffn": [
        "phase1/components/mol_ffn.py",
    ],
    "causal_recurrence": [
        "phase2/components/causal_recurrence.py",
    ],
    "zone_ed_pipeline": [
        "phase2/components/zone_e.py",
        "phase2/components/zone_d.py",
    ],
    "boundary_router": [
        "phase2/components/boundary_router.py",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def git_blob_hash(rel_path: str) -> str | None:
    """
    Return the git blob hash for a file — the SHA-1 of the file content,
    independent of commit history. Changes only when the file changes.
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


def component_file_hashes(component: str) -> dict[str, str | None]:
    """Return {file_path: blob_hash} for all files in a component."""
    return {f: git_blob_hash(f) for f in TRACKED_COMPONENTS[component]}


def load_record() -> dict:
    if VERIFIED_JSON.exists():
        with open(VERIFIED_JSON) as f:
            return json.load(f)
    return {
        "_schema_version": 2,
        "_note": "Updated by utils/verify.py update. Do not edit manually.",
        "components": {},
    }


def save_record(data: dict) -> None:
    VERIFIED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(VERIFIED_JSON, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def is_v1(record: dict) -> bool:
    return "_schema_version" not in record and "files" in record


def find_unaccounted_files() -> list[str]:
    """
    Return all .py files under COMPONENT_DIRS that are neither in
    TRACKED_COMPONENTS (any component's file list) nor in UNTRACKED_EXEMPT.

    A non-empty return value means a new component file was added without
    updating TRACKED_COMPONENTS — training should be blocked until fixed.
    """
    tracked_files: set[str] = {
        f for files in TRACKED_COMPONENTS.values() for f in files
    }
    unaccounted = []
    for dir_rel in COMPONENT_DIRS:
        dir_abs = REPO_ROOT / dir_rel
        if not dir_abs.exists():
            continue
        for path in sorted(dir_abs.glob("*.py")):
            rel = str(path.relative_to(REPO_ROOT))
            if rel not in tracked_files and rel not in UNTRACKED_EXEMPT:
                unaccounted.append(rel)
    return unaccounted


def check_component_staleness(component: str, record: dict) -> list[str] | None:
    """
    Return None if component is current, or a list of reason strings if stale.
    """
    components = record.get("components", {})
    entry = components.get(component)

    if not entry:
        return ["never verified"]

    reasons = []
    stored_hashes = entry.get("file_hashes", {})
    for f in TRACKED_COMPONENTS[component]:
        current = git_blob_hash(f)
        stored  = stored_hashes.get(f)
        if current is None:
            reasons.append(f"{f}: cannot compute hash (untracked?)")
        elif stored is None:
            reasons.append(f"{f}: not in stored record")
        elif current != stored:
            reasons.append(f"{f}: modified since {entry.get('verified_at', 'unknown')}")

    if not reasons and entry.get("result") != "pass":
        reasons.append(f"last result was '{entry.get('result')}'")

    return reasons if reasons else None


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_check(args) -> int:
    """
    Exit 0 if all checked components are current.
    Exit 1 if any are stale or have never been verified.
    """
    # Block immediately if any component files are unaccounted for.
    unaccounted = find_unaccounted_files()
    if unaccounted:
        print("=" * 60)
        print("  UNTRACKED COMPONENT FILES — training blocked")
        print("=" * 60)
        for f in unaccounted:
            print(f"  ✗  {f}")
        print()
        print("Each file in phase*/components/ must appear in either:")
        print("  TRACKED_COMPONENTS  (has a spec, will be verified)")
        print("  UNTRACKED_EXEMPT    (intentionally exempt — document the reason)")
        print("Edit utils/verify.py to add the file to one of these lists.")
        print()
        return 1

    record = load_record()

    if is_v1(record):
        print("WARNING: last_verified.json uses old v1 schema (file-level tracking).")
        print("Run: python utils/verify.py migrate")
        print("Then re-verify all components.")
        return 1

    components = args.components if args.components else list(TRACKED_COMPONENTS)

    stale = []
    for comp in components:
        if comp not in TRACKED_COMPONENTS:
            print(f"ERROR: unknown component '{comp}'. Known: {list(TRACKED_COMPONENTS)}")
            return 1
        reasons = check_component_staleness(comp, record)
        if reasons:
            stale.append((comp, reasons))

    if stale:
        print("=" * 60)
        print("  VERIFICATION STALE — training blocked")
        print("=" * 60)
        for comp, reasons in stale:
            print(f"  ✗  {comp}")
            for r in reasons:
                print(f"       {r}")
        print()
        print("Run component verification locally in Claude Code:")
        print("  Tell Claude: 'run full verification for <component>'")
        print("  Then: python utils/verify.py update --result pass --report <path> <component>")
        print()
        return 1

    print("Verification current — all checked components match last_verified.json.")
    return 0


def cmd_status(args) -> int:
    """Print a human-readable table of verification status for all tracked components."""
    unaccounted = find_unaccounted_files()
    if unaccounted:
        print("WARNING: untracked component files (not in TRACKED_COMPONENTS or UNTRACKED_EXEMPT):")
        for f in unaccounted:
            print(f"  ✗  {f}")
        print()

    record = load_record()

    if is_v1(record):
        print("WARNING: last_verified.json uses old v1 schema. Run: python utils/verify.py migrate")
        return 1

    components_record = record.get("components", {})

    col_c = 24
    col_s = 16
    col_r =  8
    col_t = 22

    header = f"{'Component':<{col_c}} {'Status':<{col_s}} {'Result':<{col_r}} {'Verified at':<{col_t}} Report"
    print(header)
    print("-" * (col_c + col_s + col_r + col_t + 30))

    for comp in TRACKED_COMPONENTS:
        entry   = components_record.get(comp, {})
        reasons = check_component_staleness(comp, record)

        if not entry:
            status = "never verified"
            result = "-"
            when   = "-"
            report = "-"
        elif reasons:
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

        print(f"{comp:<{col_c}} {status:<{col_s}} {result:<{col_r}} {when:<{col_t}} {report}")

        # Show which files are stale if relevant
        if reasons and entry:
            for r in reasons:
                print(f"  {'':>{col_c}}  {r}")

    return 0


def cmd_update(args) -> int:
    """
    Record a verification result in last_verified.json.
    Called by Claude Code after agent verification is approved.
    """
    if args.result not in ("pass", "fail"):
        print(f"ERROR: --result must be 'pass' or 'fail', got '{args.result}'")
        return 1

    if not args.components:
        print("ERROR: specify at least one component name.")
        print(f"Known: {list(TRACKED_COMPONENTS)}")
        return 1

    record = load_record()
    if is_v1(record):
        print("ERROR: last_verified.json uses old v1 schema. Run: python utils/verify.py migrate first.")
        return 1

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    updated = []
    skipped = []
    for comp in args.components:
        if comp not in TRACKED_COMPONENTS:
            print(f"ERROR: unknown component '{comp}'. Known: {list(TRACKED_COMPONENTS)}")
            skipped.append(comp)
            continue

        hashes = component_file_hashes(comp)
        missing = [f for f, h in hashes.items() if h is None]
        if missing:
            print(f"WARNING: cannot hash {missing} for '{comp}' — skipping")
            skipped.append(comp)
            continue

        record.setdefault("components", {})[comp] = {
            "file_hashes":  hashes,
            "verified_at":  now,
            "result":       args.result,
            "report":       args.report or "",
        }
        updated.append(comp)

    if updated:
        save_record(record)
        print("Updated last_verified.json:")
        for comp in updated:
            files = list(TRACKED_COMPONENTS[comp])
            print(f"  {comp}  result={args.result}  files={files}  verified_at={now}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")

    return 0 if not skipped else 1


def cmd_migrate(args) -> int:
    """
    Migrate v1 last_verified.json (file-level) to v2 (component-level).
    All components are marked never-verified — re-verify each before training.
    The old file data is preserved under '_v1_files' for reference.
    """
    if not VERIFIED_JSON.exists():
        print("No last_verified.json found — writing fresh v2 skeleton.")
        save_record({
            "_schema_version": 2,
            "_note": "Updated by utils/verify.py update. Do not edit manually.",
            "components": {},
        })
        return 0

    with open(VERIFIED_JSON) as f:
        old = json.load(f)

    if not is_v1(old):
        print("last_verified.json is already v2 — nothing to migrate.")
        return 0

    new = {
        "_schema_version": 2,
        "_note": "Updated by utils/verify.py update. Do not edit manually.",
        "_v1_files": old.get("files", {}),
        "components": {},
    }
    save_record(new)
    print("Migrated last_verified.json to v2 schema.")
    print("All components are now 'never verified' — re-verify each before training.")
    print()
    print("Components to verify:")
    for comp, files in TRACKED_COMPONENTS.items():
        print(f"  {comp}: {files}")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MoLE verification staleness checker (schema v2: component-level)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # check
    p_check = sub.add_parser("check", help="Check components are current (exit 0) or stale (exit 1)")
    p_check.add_argument("components", nargs="*", help="Components to check (default: all)")

    # status
    sub.add_parser("status", help="Print human-readable verification table")

    # update
    p_update = sub.add_parser("update", help="Record a verification result after agent run")
    p_update.add_argument("components", nargs="*", help="Component names verified")
    p_update.add_argument("--result",  required=True, choices=["pass", "fail"])
    p_update.add_argument("--report",  default="",    help="Path to the verification report")

    # migrate
    sub.add_parser("migrate", help="Migrate v1 (file-level) schema to v2 (component-level)")

    args = parser.parse_args()

    if args.command == "check":
        sys.exit(cmd_check(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "update":
        sys.exit(cmd_update(args))
    elif args.command == "migrate":
        sys.exit(cmd_migrate(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
