"""Compatibility CLI for the packaged M2 audit classifier (JSON stdin/stdout)."""
from haenv.m2.audit_classifier import (
    guess, _inject_leak, guess_lines, _ranked, _topk_score, guess_lines_topk, main,
)

if __name__ == "__main__":
    raise SystemExit(main())
