---
description: "Phase B' verification for a named component or all tracked components. /verify [component]"
---

First, run `python utils/verify.py status` and note each component's status.

**Only verify components that are `STALE` or `never verified`. Skip any component whose status is `current` — it has not changed since its last passing verification and does not need to be re-run.**

If a specific component is given, check it first with `python utils/verify.py check <component>`. If it exits 0 (current), report that it is already current and stop — do not run the Phase B' protocol for it.

For each component that does need verification: read `.claude/rules/run-full-verification.md` and execute the Phase B' protocol.

Target component: $ARGUMENTS
(If empty, verify all tracked components that are stale or never verified.)
