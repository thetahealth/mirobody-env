"""relay_quota.py -- read how much budget is left on each relay key before a batch.

An exhausted key returns 401 like a wrong key, so check balances first.
`evaluate._preflight_quota` runs the same check at start-up; this tool prints it
per key. For OneAPI-style gateways, remaining = `hard_limit_usd` (from
`/v1/dashboard/billing/subscription`) - `total_usage`/100 (from
`/dashboard/billing/usage`, in cents); both are free GET requests. A key whose
balance cannot be read is reported as unreadable, not as zero.

The billing base url and key-name prefix come from `config.backends.relay`
(`quota_url`, `key_env_pool`), normally in a local `config.local.yaml`; keys are
read from `config.env_file`. `--url` / `--prefix` override them.

Run:  uv run python tools/relay_quota.py
      uv run python tools/relay_quota.py --need 15   # also checks whether the balance covers a run

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _keys(env_file: Path, prefix: str) -> list[tuple[str, str]]:
    """`(name, value)` for every `<prefix>N=...` line in the env file, by numeric suffix."""
    out: list[tuple[str, str]] = []
    if not env_file.is_file():
        return out
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith(prefix) and "=" in line:
            k, v = line.split("=", 1)
            out.append((k.strip(), v.strip().strip('"').strip("'")))

    def idx(nv: tuple[str, str]) -> int:        # `_10` must not sort before `_2`
        t = nv[0][len(prefix):]
        return int(t) if t.isdigit() else 10 ** 6
    return sorted(out, key=idx)


def _get(base_url: str, key: str, path: str) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + path,
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http": e.code}
    except Exception as e:                                     # noqa: BLE001
        return {"_err": type(e).__name__}


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def main(argv: list[str]) -> int:
    need = _arg(argv, "--need")
    need = float(need) if need is not None else None

    sys.path.insert(0, str(ROOT))
    from haenv.cli import load_cfg
    cfg = load_cfg()
    relay = ((cfg.get("backends") or {}).get("relay") or {})
    base_url = _arg(argv, "--url") or relay.get("quota_url") or ""
    prefix = _arg(argv, "--prefix") or relay.get("key_env_pool") or "RELAY_KEY_"
    if not base_url:
        print("[quota] no billing endpoint: set config.backends.relay.quota_url "
              "(e.g. in config.local.yaml) or pass --url")
        return 2
    host = urllib.parse.urlparse(base_url).hostname or ""
    if host:
        os.environ["no_proxy"] = os.environ.get("no_proxy", "") + "," + host
        os.environ["NO_PROXY"] = os.environ["no_proxy"]

    env_file = Path(os.path.expanduser(os.path.expandvars(str(cfg.get("env_file", "")))))
    ks = _keys(env_file, prefix)
    if not ks:
        print(f"[quota] no {prefix}* keys in {env_file}")
        return 2

    print(f"env_file = {env_file} · {len(ks)} keys\n")
    print(f"{'key':16}{'limit($)':>11}{'used($)':>11}{'left($)':>11}  status")
    total, alive = 0.0, 0
    for name, key in ks:
        s = _get(base_url, key, "/v1/dashboard/billing/subscription")
        u = _get(base_url, key, "/dashboard/billing/usage")
        hl, tu = s.get("hard_limit_usd"), u.get("total_usage")
        if not isinstance(hl, (int, float)) or not isinstance(tu, (int, float)):
            print(f"{name:16}{'-':>11}{'-':>11}{'-':>11}  unreadable (usually exhausted: "
                  f"{s.get('_http') or s.get('_err')})")
            continue
        rem = hl - tu / 100.0                                  # total_usage is in cents
        total += max(0.0, rem)
        if rem > 1.0:
            alive += 1
        print(f"{name:16}{hl:>11.2f}{tu / 100.0:>11.2f}{rem:>11.4f}  "
              f"{'usable' if rem > 1.0 else 'empty'}")

    print(f"\nusable keys {alive}/{len(ks)} · total left about ${total:.2f}")
    if need is not None:
        ok = total >= need
        print(f"this run needs about ${need:.2f} => " + ("enough" if ok else "NOT enough"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
