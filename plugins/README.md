# Plugins

The main distribution registers its built-in M2/P2/P3/P4 judge entry points when installed.
`src/haenv/app/job.py` discovers a job's `plugins:` groups through
`importlib.metadata.entry_points`. The distribution shims in this directory support explicit
checkout integration; an editable installation supplies the source namespaces without putting
the repository root on `PYTHONPATH`.

| Group | Entry point | Example job |
|---|---|---|
| `haenv.m2` | `haenv.m2.plugin:judges` | `benchmarks/inputs/m2-pack1.job.yaml` |
| `haenv.pack2` | `haenv.pack2.plugin:judges` | `benchmarks/inputs/pack2-triage.job.yaml` |
| `haenv.p2` | `haenv.p2.plugin:judges` | The registered P2 contract. |
| `haenv.p3` | `haenv.p3.plugin:judges` | `benchmarks/inputs/p3-meds.job.yaml` |
| `haenv.p4` | `haenv.p4.plugin:judges` | `benchmarks/inputs/p4-followup.job.yaml` |

From an installed checkout, use the public command for a deterministic build:

```sh
uv run haenv build benchmarks/inputs/m2-core.job.yaml --gen deterministic --fresh
```

Pack and view declarations live in `src/haenv/contracts/pack.py`; registration and explicit
application hooks live in `src/haenv/app/packs.py`. The shared emission and batch audit is in
`src/haenv/world/validation/shared_audit.py`. A batch's passing audit marker remains required
before evaluation; the command is `tools/validation/pack_audit.py`.

The [documentation index](../docs/README.md) explains package responsibilities and the
[external task contract](../docs/design/external-task-contract.md) describes extensions.
