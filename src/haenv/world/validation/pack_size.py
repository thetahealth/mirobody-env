"""Pack size and seed: how many items a plugin pack publishes and which seed orders the selection.

A pack job carries a `pack` header `{n_items, seed_sha256}`; the shipped job of the public sample
pack also carries its `seed`. The official board's seed stays private; its batch records only the
seed's hash; when a board is retired its seed is published for reproduction.

* `apportion` turns a pack's proportion table into item counts for any N (largest remainder);
* `split_h` halves each class count over the hidden-disease flag H, keeping the H totals equal;
* `seeded` salts a selection tag with the seed (the empty public seed leaves the tag as it is);
* `variant_base` gives a non-public seed its own window of variant numbers (new patients per seed);
* `pool_n` / `nest_chain` make packs nested: the pool is composed for the largest size of `NEST_SIZES`
  (or N when larger) and a pack of N extends the pack of every smaller nested size, so the items of a
  50-item run stay valid, byte for byte, when the 100-item pack of the same seed is run.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import pathlib

#: seed of the public sample pack (shipped job files)
PUBLIC_SEED = ""
#: item count of the shipped job files
DEFAULT_N = 50
#: a pack generator emits at least this many times N candidate items; the audit's shortcut gates read
#: that pool and the pack's `select` publishes N of them (`shared_audit` SA-0 holds the line)
POOL_FACTOR = 4


#: Nested pack sizes: same seed, same pool, and the pack of a smaller size is a subset of the pack of a
#: larger one (a run of the smaller pack is extended, not repeated, when the larger one is run).
NEST_SIZES = (50, 100)


def pool_n(n: int) -> int:
    """The pack size a generator composes its pool for: the largest nested size, or n when larger."""
    return max(int(n), max(NEST_SIZES))


def nest_chain(n: int) -> list[int]:
    """The sizes a selection of n extends in order: the nested sizes below n, then n."""
    return [k for k in NEST_SIZES if k < int(n)] + [int(n)]


def pool_target(n: int) -> int:
    """Smallest pool a pack of `n` items must emit."""
    return POOL_FACTOR * int(n)


def seed_sha256(seed: str) -> str:
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()


def seeded(tag: str, seed: str) -> str:
    return f"{tag}{seed}"


#: Width of a seed's variant window (`variant_base`): a pool instantiates at most this many variant
#: numbers of one condition. The public pools use variants 1-60 at N = 100 and up to about 210 at N = 400.
VARIANT_WINDOW = 1000


def variant_base(seed: str) -> int:
    """The variant number just before a seed's window: 0 for the public seed (variants from 1, as
    shipped), else a multiple of `VARIANT_WINDOW` in [VARIANT_WINDOW, 999 x VARIANT_WINDOW] drawn by
    sha256 of the seed (salt `variant-window`), so a window never meets the public variants. A generator
    that instantiates its patients from the window gives a new seed new patients (every per-case world
    draw is keyed by the case id) while its composition table stays as it is. The window number carries
    about 10 bits of the seed's hash and is published with the case ids; the seed stays private."""
    if seed == PUBLIC_SEED:
        return 0
    k = int(hashlib.sha256(f"variant-window|{seed}".encode("utf-8")).hexdigest()[:12], 16) % 999
    return VARIANT_WINDOW * (1 + k)


def cli_seed(arg: str | None) -> str:
    """A generator's seed: the `--seed` argument, else HAENV_PACK_SEED, else the public seed."""
    import os
    return arg if arg is not None else os.environ.get("HAENV_PACK_SEED", PUBLIC_SEED)


def header(n_items: int, seed: str) -> dict:
    """The job's `pack` header; the seed itself is written only when it is the public one."""
    out = {"n_items": int(n_items), "seed_sha256": seed_sha256(seed)}
    if seed == PUBLIC_SEED:
        out["seed"] = seed
    return out


def apportion(n: int, weights: dict) -> dict:
    """Largest-remainder apportionment of n over weights; ties go to the larger weight, then to the
    earlier key in declaration order."""
    keys = list(weights)
    tot = sum(weights.values())
    if n < 0 or (tot <= 0 and n):
        raise ValueError(f"cannot apportion {n} over {weights}")
    exact = {k: n * weights[k] / tot if tot else 0.0 for k in keys}
    out = {k: int(exact[k]) for k in keys}
    rest = n - sum(out.values())
    order = sorted(keys, key=lambda k: (-(exact[k] - out[k]), -weights[k], keys.index(k)))
    for k in order[:rest]:
        out[k] += 1
    return out


def split_h(n_by_class: dict) -> dict:
    """{(H, class): n}: each class count halved over H = 1 and H = 0; an odd count gives its extra
    item to H = 1 and H = 0 in turn (class order), so the H totals differ by at most one."""
    out, turn = {}, 1
    for c, n in n_by_class.items():
        lo = n // 2
        out[(1, c)] = lo + (n % 2 if turn == 1 else 0)
        out[(0, c)] = lo + (n % 2 if turn == 0 else 0)
        if n % 2:
            turn = 1 - turn
    return out


def short(need: dict, have: dict, what: str) -> None:
    """Raise when a cell's supply is below its quota, naming every gap (never shrink silently)."""
    gaps = {k: (need[k], have.get(k, 0)) for k in need if have.get(k, 0) < need[k]}
    if gaps:
        msg = "; ".join(f"{k}: need {n}, have {h}" for k, (n, h) in sorted(gaps.items(), key=str))
        raise SystemExit(f"[pack] {what}: pool too small -- {msg}")


def read_header(job: dict | str | pathlib.Path | None) -> dict:
    """The `pack` header of a job (a parsed mapping or a path); {} for a job without one."""
    if job is None:
        return {}
    if not isinstance(job, dict):
        from haenv.resources.yamlcache import load_yaml
        job = load_yaml(job) or {}
    return dict(job.get("pack") or {})


def resolve_seed(job, seed: str | None) -> str:
    """The seed a selection uses: `seed` when given (it must match the job's recorded hash), else
    the seed the job carries (the public sample)."""
    h = read_header(job)
    if seed is None:
        if "seed" not in h:
            raise SystemExit("[pack] this job records only its seed's hash: pass --seed")
        seed = str(h["seed"])
    if h.get("seed_sha256") and seed_sha256(seed) != h["seed_sha256"]:
        raise SystemExit("[pack] --seed does not match the job's seed_sha256")
    return seed


def batch_fields(job_path: str | pathlib.Path | None, job_id: str) -> dict:
    """What a built batch records about its pack: N, the seed's hash and the job, never the seed."""
    try:
        h = read_header(job_path) if job_path else {}
    except (OSError, ValueError):
        return {}
    if not h:
        return {}
    return {"n_items": h.get("n_items"), "seed_sha256": h.get("seed_sha256"), "job": job_id}
