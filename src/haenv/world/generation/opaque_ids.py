"""Opaque solver-facing case ids (world 1.4 id-leak fix, prototype).

The real case id (`JD-04v15`, `HD-...`) is the world's RNG key and the verifier's lookup key,
but its shape and variant number correlate with gold and difficulty in identifier-only
shortcut checks. The solver therefore sees an opaque id instead:

- `solver_case_id(cid, seed)` is `C` + 10 hex digits of HMAC-SHA256(seed, cid). It is
  deterministic from (seed, real id), has a fixed shape and length, and without the seed it
  cannot be inverted to the real id or to its variant number.
- `build.py` stores it in the verifier-only `latent_premise.meta.solver_case_id` (the
  `meta.*` subtree is exempt bookkeeping, `gates_case.PREMISE_REGISTRY`) and names the
  evidence `EV-<opaque>-NN`. `haenv_kernel.build.build_instance` puts it in the solver
  payload; the verifier payload keeps the real id, so `wq.injected_manifest(vp.case_id)`
  resolves as before.

The judging code is frozen (`haenv.anchor.judging_files`). A few judging functions look a
table up by the *solver* payload's `case_id` (`qside` prompt suffixes, `baselines` stubs,
`gates_outcome._real_symptom_evidence`). This module keeps those lookups resolving without
editing them: the Q-side manifest is also registered under the opaque id
(`register_manifest_alias`), and the run context's per-case tables resolve an opaque key to
the real one (`alias_run_context`).

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import hashlib
import hmac

#: Where the opaque id lives in `RawCase.latent_premise` (verifier-only).
META_KEY = "solver_case_id"

#: Opaque id -> real id, for this process (filled by `remember`).
_REAL: dict[str, str] = {}


def solver_case_id(case_id: str, seed: str | None = None) -> str:
    """`C` + 10 upper-case hex digits of HMAC-SHA256(key=seed, msg=case_id)."""
    if seed is None:
        from haenv.world.validation.pack_size import cli_seed
        seed = cli_seed(None)
    mac = hmac.new(str(seed).encode("utf-8"), f"solver-case-id|{case_id}".encode("utf-8"),
                   hashlib.sha256).hexdigest()
    return "C" + mac[:10].upper()


def stored_solver_id(raw) -> str | None:
    """The opaque id recorded on a built case, or None for a case built before the fix."""
    lp = getattr(raw, "latent_premise", None) if not isinstance(raw, dict) else raw.get("latent_premise")
    sid = ((lp or {}).get("meta") or {}).get(META_KEY)
    return str(sid) if sid else None


def assign(raw, seed: str | None = None) -> str:
    """Record the opaque id on `raw` (in place) and return it."""
    sid = solver_case_id(str(raw.case_id), seed)
    lp = dict(raw.latent_premise or {})
    lp["meta"] = {**(lp.get("meta") or {}), META_KEY: sid}
    raw.latent_premise = lp
    remember(str(raw.case_id), sid)
    return sid


def remember(case_id: str, sid: str | None) -> None:
    if sid and sid != case_id:
        _REAL[str(sid)] = str(case_id)


def remember_built(built: dict) -> dict[str, str]:
    """Remember every opaque id of `built` ({cid: RawCase}); returns {opaque: real}."""
    out = {}
    for cid, raw in (built or {}).items():
        sid = stored_solver_id(raw)
        if sid and sid != str(cid):
            out[sid] = str(cid)
    _REAL.update(out)
    return out


def real_case_id(cid) -> str:
    """The real id behind an opaque one; any other id comes back unchanged."""
    return _REAL.get(str(cid), str(cid))


def register_manifest_alias(case_id: str, sid: str | None) -> None:
    """Register the real id's Q-side manifest under the opaque id too, so a lookup keyed by
    the solver payload's id (`gates_outcome._real_symptom_evidence`) finds it."""
    import haenv.world.wq as wq
    if not sid or sid == case_id:
        return
    man = wq.injected_manifest(case_id, required=False)
    if man:
        wq.register_injection(sid, man)


class AliasDict(dict):
    """A per-case table keyed by the real id that also answers for the opaque id.
    Iteration, `len` and copies see the real keys only."""

    def _key(self, k):
        return k if dict.__contains__(self, k) else _REAL.get(str(k), k)

    def get(self, k, default=None):
        return dict.get(self, self._key(k), default)

    def __getitem__(self, k):
        return dict.__getitem__(self, self._key(k))

    def __contains__(self, k):
        return dict.__contains__(self, self._key(k))


#: RunContext tables that judging code reads by the solver payload's `case_id`.
CTX_TABLES = ("premise_for", "noop_for", "quant_for", "oracle_gold_tests",
              "oracle_disc_names", "oracle_diagnosis", "oracle_warranted")


def alias_run_context(ctx, built: dict) -> int:
    """Make the run context's per-case tables answer for opaque ids; returns how many
    cases carry one."""
    n = len(remember_built(built))
    for name in CTX_TABLES:
        tab = getattr(ctx, name, None)
        if isinstance(tab, dict) and not isinstance(tab, AliasDict):
            setattr(ctx, name, AliasDict(tab))
    return n
