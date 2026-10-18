"""rng.py -- counter-based deterministic randomness.

Each value is a pure function of its path, `f(path) -> u32`, so adding a draw
for one case or field never changes any other value (a sequential
`random.Random(seed)` would shift every later draw).

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import hashlib


def counter_u32(*path) -> int:
    """path -> a deterministic integer in [0, 2^32). Changing any segment of `path` yields an
    independent value."""
    key = "|".join(str(p) for p in path).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def unit(*path) -> float:
    """path -> a deterministic float in [0, 1)."""
    return counter_u32(*path) / 2 ** 32


def below(n: int, *path) -> int:
    """path -> a deterministic integer in [0, n). Returns 0 when n <= 0 (callers need not
    check first)."""
    return counter_u32(*path) % n if n > 0 else 0


def pick(seq, *path):
    """Deterministically pick one item from an ordered sequence; sets are rejected
    because their iteration order is unstable.
    """
    if isinstance(seq, (set, frozenset)):
        raise TypeError("pick() does not accept a set: set iteration order is unstable and breaks reproducibility; pass a list or tuple")
    items = list(seq)
    if not items:
        raise ValueError("pick() got an empty sequence")
    return items[below(len(items), *path)]


def subset(seq, k_lo: int, k_hi: int, *path) -> list:
    """Deterministically pick a subset of size in [k_lo, k_hi], preserving the
    input order so downstream sequential generation does not depend on the draw.
    """
    items = list(seq)
    if isinstance(seq, (set, frozenset)):
        raise TypeError("subset() does not accept a set (same reason as pick)")
    k_hi = min(k_hi, len(items))
    k_lo = max(0, min(k_lo, k_hi))
    k = k_lo + below(k_hi - k_lo + 1, *path, "k")
    # Independent per-item scores: adding a candidate never reorders the others.
    scored = sorted(range(len(items)), key=lambda i: counter_u32(*path, "item", items[i]))
    keep = sorted(scored[:k])
    return [items[i] for i in keep]


def day_jitter_compat(case_id: str, kind: str, i: int, amp: int) -> int:
    """Reproduces `events._day_jitter` bit for bit (digest_size=8, same key format),
    so `events` can move onto this module without changing any value.
    """
    h = hashlib.blake2b(f"{case_id}|{kind}|{i}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 * amp + 1) - amp
