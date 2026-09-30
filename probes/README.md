# probes/ -- question framing (`D_probe`)

Question difficulty has two orthogonal knobs: `D_case` (the case content) and
`D_probe` (how the question is asked). This directory holds the second one.
Comparability across models requires that every model see the same prompt
within a given cell; different cells may use different framings.

Each probe names a registered prompt template through `framing_ref` (the
registry lives in `haenv/framings.py`) and pins it with `framing_sha256`.
`load_probes` in `haenv/evaluate.py` rejects a probe whose template is not
registered or whose fingerprint does not match, so an edit to a template
fails loudly instead of silently changing both arms of an experiment.

Rules:
- one file per probe, with a globally unique `probe_id`;
- `framing_ref` and `framing_sha256` are required;
- `answer_space` must be declared -- judges are wired up based on it
  (`driver_vocab` uses the C track, `free` uses the dx judges);
- a new probe must render a prompt different from every existing probe;
  otherwise it adds a dimension that cannot be tested.
