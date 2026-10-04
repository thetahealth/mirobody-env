# plugins/

Entry-point shims for judge groups whose code lives in this repo. `haenv/job.py` loads a job's
`plugins:` groups through `importlib.metadata.entry_points`, which only sees installed
distributions; a `*.dist-info` directory on `sys.path` is one, so putting this directory on
`PYTHONPATH` makes the group discoverable without installing anything into a shared venv.

| group | entry point | used by |
|---|---|---|
| `haenv.m2` | `haenv.m2.plugin:judges` | `inputs/m2-core.job.yaml`, `inputs/m2-pack1.job.yaml` |
| `haenv.pack2` | `haenv.pack2.plugin:judges` | `inputs/pack2-triage.job.yaml` |
| `haenv.p3` | `haenv.p3.plugin:judges` | `inputs/p3-meds.job.yaml` |
| `haenv.p4` | `haenv.p4.plugin:judges` | `inputs/p4-followup.job.yaml` |

    PYTHONPATH=<repo>:<repo>/plugins python -m haenv build inputs/m2-core.job.yaml --gen deterministic --fresh

Each group registers through the shared skeleton (`haenv/pack_skeleton.py:register_pack`) and its
emission gate is the shared one (`haenv/shared_audit.py`, SA-1 + SA-8) on the pack's declaration. An
emitted batch is published only after the shared batch audit passes, which writes the audit marker
that `haenv run` requires on a pack's cells:

    PYTHONPATH=<repo> python tools/pack_audit.py --pack p4 --batch <batch> --out <dir>
    ... --pack pack2 --job inputs/pack2-triage.job.yaml
    ... --pack m2 --job inputs/m2-pack1.job.yaml --ref <v1.0.1 deterministic ddx-workup batch>
