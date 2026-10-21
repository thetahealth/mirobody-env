"""The deterministic example world used by the idempotency controls."""
import os
from pathlib import Path
import subprocess
import sys


def example_batch(cache):
    root = Path(__file__).resolve().parents[1]
    def build(dest):
        env = dict(os.environ, HAENV_OUTPUT_ROOT=str(dest / "out"),
                   PYTHONPATH=os.pathsep.join(map(str, (root / "src", root, root / "plugins"))))
        result = subprocess.run([sys.executable, str(root / "tests/_build_cli.py"), "build",
                                 "benchmarks/inputs/example-ew.job.yaml", "--gen", "deterministic", "--fresh"],
                                cwd=root, env=env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr[-2000:]
    return cache.directory("idempotency-source", {}, build)
