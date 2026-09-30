# The `multi` geometry: judge mounting and the trajectory subject

The multi-round geometry (`multi`) runs one case as a sequence of rounds: the model answers,
time advances, new observations arrive, and the model answers again. Its judges run through
`judges.run_judges` like every other geometry.

## Subjects on `multi`

| Subject | What it holds | Judges |
|---|---|---|
| `LAST` | the final round's answer (`ctx["round_outputs"][-1]`) | the 17 single-answer judges: `forecast` `driver` `alternative` `dx_unified` `dx_comorbidity` `dx_independent` `join_type` `dx_rival` `disc_tool` `join_selfcheck` `join_cover` `workup` `abstention` `commit_timing` `review_flag` `self_contradictory_exclusion` `what_not_to_do` |
| `TRAJ` | the episode: one dict per round with `round · day · risk_cat · risk · top_driver · action · repair · visible_ev · cited_ev · n_new_points · notes` (`mount_table.TRAJ_ROW_FIELDS`) | `multiround_revision`, `premise_repair` |

If the round outputs are missing, the `LAST` judges are recorded under
`mount_subject_missing`; the trajectory is never substituted, because it has a different
shape. `premise_repair` reads its premise from `ctx["premise"]` through the adapter
`_j_premise_repair` and returns `{}` (not applicable) without one.

The 8 slice-only judges (`slices*`, `slice_revision`) are not mounted on `multi`: they read
fields a trajectory row does not have. Their multi-round counterpart is
`multiround_revision`. Every empty cell has a reason in `mount_table.WHY_NOT`.

Cell values outside `SUBJECTS` raise, both at import and on every lookup.

## Plugging in an external judge

A judge that reads the trajectory registers with `register_judge()` and mounts with
`mount_table.mount(name, {"multi": TRAJ}, why_not=…)`.

A judge that needs a differently shaped episode registers its own view with
`mount_table.register_subject(name, build, source=..., why=..., row_fields=...)`.
`run_judges` calls `build(ctx)` and passes the result to the judges mounted on that
subject; if `build` returns `None`, the judge is skipped and the row records it. The four
built-in subjects cannot be overridden. `examples/external_subject_demo/` is a complete example.

An external judge must also:

1. accept `fn(subject, vp, ctx)`, with an adapter if its own signature differs;
2. declare the gold kinds it applies to (`("*",)` for all; a new gold shape needs
   `gold_kinds.register_kind_rule`);
3. have a role in `registry/scoring*.yaml` to count toward the headline score. Without
   one it is `diagnostic`: reported, not scored.

`TRAJ` carries only the fields listed above; its free-text field is `notes`, truncated to 600
characters. A judge that needs anything else registers its own subject.

---

*SYNTHETIC data, evaluation use only, not medical advice.*
