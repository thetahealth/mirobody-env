# Frozen question material

The question-half JSONL files, `MANIFEST.json` and `CANARY.txt` are retained as one original
set. The manifest identifies the packs, Q filenames, counts and hashes; read those declarations
directly. The canary remains outside the case body, and Q content stops at the permitted index
time.

The manifest's `job` paths, freeze revision and pack hashes are original provenance labels.
For example, an original `inputs/...` label does not become a new measurement because its job
now lives under `benchmarks/inputs/`. Consumers must distinguish the original label from the
current resource locator. Do not edit the manifest or Q rows to hide a path migration.

The private truth halves and complete case files are not supplied by a clean checkout. Their
absence is not an empty valid pack, and they must not be invented. Full split/join verification
requires the original protected inputs and the corresponding implementation. Keep generated
truth halves and new reconstruction output subject to their original distribution restrictions.

The importable split/join implementation is maintained separately from this data directory.
Moving the data alone does not relocate or reconfigure that implementation.
