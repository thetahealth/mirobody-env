"""The deterministic example world used by the idempotency controls."""
import os
from pathlib import Path
import subprocess
import sys

from haenv.app.config import load_cfg
from haenv.i18n import resolve
from haenv.resources.yamlcache import load_yaml
from haenv.world.validation.pack_size import cli_seed, seed_sha256


def example_batch(cache):
    root = Path(__file__).resolve().parents[1]
    def build(dest):
        env = dict(os.environ, HAENV_OUTPUT_ROOT=str(dest / "out"),
                   PYTHONPATH=os.pathsep.join(map(str, (root / "src", root, root / "plugins"))))
        result = subprocess.run([sys.executable, str(root / "tests/_build_cli.py"), "build",
                                 "benchmarks/inputs/example-ew.job.yaml", "--gen", "deterministic", "--fresh"],
                                cwd=root, env=env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr[-2000:]
    language = resolve(job=load_yaml(root / "benchmarks/inputs/example-ew.job.yaml"), cfg=load_cfg())
    inputs = {"language": language, "seed_sha256": seed_sha256(cli_seed(None))}
    return cache.directory("idempotency-source", inputs, build)
