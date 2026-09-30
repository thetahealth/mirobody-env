<!--
Thanks for sending a PR. CONTRIBUTING.md explains the checks below; read it first.
-->

## What this changes

<!-- One or two sentences. What was wrong, and what is different now. -->

## Why

<!-- The failure this fixes, or the gap it fills. If a reading motivated it, paste the
     reading rather than describing it. -->

## Readings

<!-- Paste actual command output, not a summary. "It works" is not a reading. -->

```
$ uv run python tools/verify_selftest.py
$ uv run python tools/leak_probe_selftest.py
```

---

## Checklist

- [ ] Both self-checks above exit 0, and the four Quick Start commands in README.md exit 0
- [ ] `python -m compileall haenv verifier_core` passes on **3.10 and 3.12**
      (`requires-python = ">=3.10"` — a syntax feature newer than that breaks the promise)
- [ ] If I touched scoring code, I re-anchored the segment fingerprints:
      `uv run python tools/make_freeze.py --revision <n+1> --segments-only --supersedes <n> --why "…"`
      — this needs **no evaluation artifacts** and takes about two seconds
- [ ] If I added a check, it comes **with a counter-example**: one input proving it fires,
      one proving it is not firing indiscriminately
- [ ] No absolute paths from my machine, no credentials, no real patient data

## Anything you deliberately did not do

<!-- Known gaps are welcome here and count in your favour. Silent gaps do not. -->
