"""Pure route configuration contracts shared by schemas and live transports.

No configuration loading, scheduler setup or network patching on import.
"""
from __future__ import annotations


class ConfigError(ValueError):
    """The merged configuration does not match the schema."""


def check_proxy(value) -> str:
    if value == "none":
        return "direct"
    if value not in ("env", "direct"):
        from urllib.parse import urlsplit
        parsed = urlsplit(value) if isinstance(value, str) else None
        if (parsed is None or parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("proxy must be env, none/direct, or an HTTP(S) URL without credentials")
    return value


def validate_proxy_config(cfg: dict, validate) -> dict:
    """Validate transport-only extensions without changing the frozen world schema."""
    import copy
    raw = copy.deepcopy(cfg)
    routes = {}
    for section in ("models", "backends"):
        entries = raw.get(section)
        if not isinstance(entries, dict):
            continue
        for name, spec in entries.items():
            if isinstance(spec, dict) and "proxy" in spec:
                try:
                    routes[section, name] = check_proxy(spec.pop("proxy"))
                except ValueError as error:
                    raise ConfigError(f"{section}.{name}.{error}") from None
    out = validate(raw)
    for (section, name), value in routes.items():
        if section == "backends":
            out.setdefault("transport", {}).setdefault("backend_proxies", {})[name] = value
        else:
            out[section][name]["proxy"] = value
    for value in (out.get("transport", {}).get("backend_proxies") or {}).values():
        check_proxy(value)
    return out
