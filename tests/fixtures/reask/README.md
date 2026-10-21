This compressed fixture contains stored synthetic diagnosis conversations, scorer payloads,
recorded purchase observations, ordered result candidates and reference arm manifests.
The tests unpack it locally; no model or network access is needed. One unit covers an odd
number of missed results; another covers a decisive archived record returned by a read tool.

Tool content and candidate order are preserved. Source provenance strings are replaced by
`recorded`; response rows retain only the final-request fingerprint needed for verification.
The expected request-envelope hashes were compared with the archived runner's pure wrapper.
`facts.json` records the measured full-corpus offline comparison; that larger corpus is not
required by the portable tests. Live answers, provider wire formats and billing metadata are
outside this fixture's equivalence assertions.
