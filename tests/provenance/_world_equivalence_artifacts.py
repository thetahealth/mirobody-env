"""The release comparison regenerates each frozen group once per content key."""
from pathlib import Path
import os
import subprocess
import sys
from tools.validation.world_equivalence import ROOT, proof, job_output, sha


def specifications():
    data = proof()
    return [("job", row["id"]) for row in data["jobs"]] + [("world", pack) for pack in data["worlds"]]


def regenerate(cache, spec):
    kind, name = spec
    data = proof()
    def build(dest):
        Path(__file__).read_bytes()
        if kind == "job":
            row = next(row for row in data["jobs"] if row["id"] == name)
            (dest / "job.yaml").write_text(job_output(row["task"]))
        else:
            env = dict(os.environ, HAENV_OUTPUT_ROOT=str(dest), UV_NO_SYNC="1", UV_FROZEN="1")
            with (dest / "build.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT / "tools/validation/world_equivalence.py"),
                                "--build", name], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    dest = cache.directory("world-equivalence", {"kind": kind, "id": name}, build)
    if kind == "job":
        return {"job.yaml": sha(dest / "job.yaml")}
    files = {}
    for filename in ("cases.jsonl", "payloads.jsonl"):
        path, = dest.glob(f"results/*/*/*/{filename}")
        files[filename] = sha(path)
    return files
