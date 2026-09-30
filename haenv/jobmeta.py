"""What a job file declares about itself, read from the yaml and never from the file name.

`task_type` names the `results/<task_type>/` directory a job's batches live in; `gated` marks the
tool-track arm. Tools that group or locate packs go through here so a renamed job (any name that
does not start with its task type) is still found.
"""
from __future__ import annotations

from pathlib import Path

from .job import TASK_TYPES
from .yamlcache import load_yaml


def job_meta(path: str | Path) -> tuple[str, bool]:
    """(task_type, gated) of a job file; `task_type` defaults as `load_job` defaults it."""
    data = load_yaml(Path(path))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a job file must be a YAML mapping")
    task_type = data.get("task_type", "early_warning")
    if task_type not in TASK_TYPES:
        raise ValueError(f"{path}: task_type={task_type!r} is not in {TASK_TYPES}")
    return task_type, bool(data.get("gated", False))


def pack_task_type(root: Path, job: str, batch: str) -> str:
    """The `results/<task_type>/` directory that holds batch `batch` of job `job`.

    Read from `inputs/<job>.job.yaml`; without that file, from where the batch directory is (exactly
    one task type may hold it). Never inferred from the job name; unfindable is an error.
    """
    jy = Path(root) / "inputs" / f"{job}.job.yaml"
    if jy.is_file():
        return job_meta(jy)[0]
    hits = sorted(p.parent.parent.name for p in (Path(root) / "results").glob(f"*/{job}/{batch}")
                  if p.is_dir())
    if len(hits) != 1:
        raise ValueError(f"cannot tell the task type of job {job!r}: no {jy} and {len(hits)} "
                         f"results/*/{job}/{batch} directories")
    return hits[0]
