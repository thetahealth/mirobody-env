"""Read regression tables while preserving explicit tuple and set values."""
import json


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"tuple"}:
            return tuple(_decode(item) for item in value["tuple"])
        if set(value) == {"set"}:
            return {_decode(item) for item in value["set"]}
        return {key: _decode(item) for key, item in value.items()}
    return value


def load_rows(path, name):
    return _decode(json.loads(path.read_text())[name])
