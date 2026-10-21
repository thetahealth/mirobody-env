"""A requested judge effort must be present in the actual provider payload."""

import io

import json

import pytest

from haenv.app.evaluate import OpenAICompatSolver

class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


__all__ = [name for name in globals() if not name.startswith("__")]
