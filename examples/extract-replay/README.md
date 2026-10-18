# Offline extraction replay

This is a formal synthetic example for the clinician-gold extraction path. The
[case text](clinician-placeholder/rc-placeholder-01.md) and
[annotation placeholder](clinician-placeholder/annotations.placeholder.json) are handwritten
placeholders. They are not clinician judgements and must not supply benchmark readings.

The checked-in replay JSON contains handwritten extracted patient facts and explicitly records
`origin: placeholder-handwritten`. Its name is the original hash of the model, arguments and
prompt. The case text and replay bytes are unchanged. In particular, `source_text` inside the
JSON keeps its historical source path; that label is not a current file locator.

The runtime extraction reader must resolve replays here. With `--extract-offline`, a missing
matching replay or cache entry is an explicit error and must not trigger a model request.
The [reproduction guide](../../docs/REPRODUCE.md) describes extraction and gold provenance;
the [product specification](../../docs/spec/spec.md#2-visible-records-and-verifier-only-truth)
defines the solver-visible boundary. Relocation does not establish clinical validity.
