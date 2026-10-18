# Freeze and provenance resources

[`freeze-2026-09-02.json`](freeze-2026-09-02.json) is the generated source and pack
declaration. Read its `revision`, segment declarations and pack entries directly; the date
in its stable filename does not identify the current source revision. Source-only registrations
can inherit pack objects, whose original generation identity remains authoritative.

[`CANARY.txt`](CANARY.txt) is the contamination canary sidecar. Both files are formal
resources, included with their runtime readers and distribution manifests. Moving them does
not authorize a new freeze, new pack generation, rescoring or editing a historical stamp.

See the [current provenance contract](../../docs/spec/spec.md#11-criterion-provenance-and-publication)
and [anchor guide](../../docs/anchor/README.md). Measurement-validity data lives separately in
[`../validity/`](../validity/).
