# P4 visible change and cause attribution

P4 distinguishes true change, method difference, preanalytical effects and
analytical or biological noise. A non-noise item must show a change beyond the
paired reference change value (RCV) in the printed reports, including the
applicable method correction or standardized specimen verification. The RCV
calculation and clinical sources live in `registry/labworld.yaml`; the audit
thresholds live in `registry/difficulty_r2.yaml`.

The `p4_magnitude_overlap` admission gate measures whether report-header amplitude
alone identifies the cause among the three non-noise classes. It uses the existing
depth-one tree, five folds, random seed and chance-corrected threshold of 0.20.
Separated amplitude ranges among these causes must fail the gate. Noise versus
significant change is the clinical first step, so that distinction alone does not
fail this cause-attribution gate. The all-class amplitude reading remains reported.

This scope is a haenv-authored measurement policy, not a new clinical threshold.
Clinical interpretation still uses the complete visible record and the registered
P4 rules. Changing a documentation source label changes source provenance; it
does not authorize relabelling a previously generated pack.
