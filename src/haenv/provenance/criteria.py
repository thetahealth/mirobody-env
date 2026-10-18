"""Formal criterion entry points and dispatch boundaries.

Historical readers interpret the declarations recorded by the selected revision.
"""

# Roots describe scoring entry points, never their dependency lists. Pack
# registrations contain both world callbacks and judges; follow the registered
# scoring callbacks rather than treating a whole frozen world/registration
# segment as an entry point. Files are still hashed in full (conservative).
PACK_SCORING = {
    "shared_roots": ('haenv/app/evaluate.py', 'haenv/app/packs.py', 'haenv/scoring/legacy.py'),
    "shared_symbol_roots": (('haenv/interaction/query/__init__.py', ("_armed_row",)),),
    "import_scope": "symbols",
}

# Outgoing imports inside these dispatch functions are registration edges, not
# dependencies of every possible consumer. The dispatcher file is hashed and
# each stopped edge is recorded. An import elsewhere, even into another pack,
# is never stopped merely because of its destination.
DISPATCH_BOUNDARIES = {
    ("haenv/interaction/query/solver.py", "QueryMixin.solve"):
        "Selects single/query versus the M2 gated geometry; M2 roots its gated-round implementation explicitly, while every criterion retains the shared query guard and row executor.",
    ("haenv/interaction/query/__init__.py", "install"):
        "Registers requested query adapters; actual execute_row/_armed_row and its guards remain criterion dependencies.",
    ("haenv/world/history/history_dependency.py", "install"):
        "Imports and registers the selected history pack's callbacks; each pack criterion roots its actual judge and row execution, not every pack world.",
    ('haenv/app/packs.py', "register_pack"):
        "Registers a selected Pack's world callbacks and judge; each criterion roots its own judge explicitly.",
    ('haenv/app/packs.py', "install"):
        "Bootstraps shared world/evaluation wrappers through the pack registry; evaluate.py is an explicit shared scoring root.",
    ("haenv/judges/__init__.py", "load_judge_plugins"):
        "Loads the requested entry-point group into the judge registry; registrations do not make every plugin a scoring dependency.",
    ('haenv/interaction/query/adapter.py', "load_pack_adapters"):
        "Imports every pack's query adapter solely for registration; each criterion roots its own adapter and shared profile/guard path.",
    ('haenv/app/job.py', "_load_judge_groups"):
        "Dispatches job-declared plugin groups; each pack's actual judging entry is rooted independently.",
    ('haenv/world/protocols/delivery.py', "question"):
        "Dispatches solver-visible question templates by pack; those templates are world inputs, not saved-answer scoring calls.",
    ('haenv/world/clinical/med_course.py', "overlay_medicines"):
        "Dispatches world-medication generators by pack; named medication helpers used by judges remain ordinary dependencies.",
}

CRITERIA = {
    "kernel": {"roots": ("haenv_kernel/verifier.py", 'haenv/scoring/gates.py'),
               "optional_symbol_roots": (("haenv_kernel/__init__.py", ("grade",)),),
               "fields": ("gates", "overall", "track_*", "tracks")},
    "diagnosis": {"roots": ("haenv/judges/differential.py", "haenv/judges/outcome.py"),
                  "fields": ("dx_*", "join_*", "disc_*", "workup_*", "direction_ok", "driver_*", "alternative_*")},
    "trajectory": {"roots": ("haenv/judges/trajectory.py", 'haenv/scoring/process.py'),
                   "fields": ("noop_*", "abst_*", "quant_*", "premise_*", "slice_*", "commit_*", "revision_*")},
    "safety": {"roots": ("haenv/judges/safety.py",),
               "fields": ("review_*", "self_contradictory_*", "what_not_to_do*", "safety_*")},
    "llm_rubric": {"roots": ('haenv/scoring/llm_rubric.py', "haenv/judges/llm.py"),
                   "fields": ("llm_*", "rival_discriminator*")},
    "semantic": {"roots": ('haenv/scoring/semantic/pipeline.py', 'haenv/scoring/semantic/report.py'),
                 "fields": ("semantic", "semantic_*",)},
    "management_rubric": {"roots": ('haenv/scoring/management/pipeline.py',),
                          "optional_symbol_roots": (("haenv/scoring/routes.py",
                                                     ("RoutedJudges", "configured_routes")),),
                          # The delta merger also returns scores. It did not
                          # exist in early archives, whose identity remains reconstructible.
                          "optional_roots": ('haenv/scoring/management/precision_rejudge.py',),
                          "fields": ("management_*",)},
    "scoring": {"roots": ('haenv/scoring/legacy.py', "haenv/analytics.py", 'haenv/scoring/rows.py', "haenv/report.py", 'haenv/scoring/ranking.py'),
                "fields": ("score", "core_*", "overall", "gold_kind")},
    "P2_JUDGING": {**PACK_SCORING,
                   "roots": ("haenv/p2/score.py",),
                   "judge_registry": "haenv/p2/plugin.py",
                   "optional_symbol_roots": (("haenv/p2/query.py", ("ADAPTER",)),),
                   "fields": ("p2_*",)},
    "P3_JUDGING": {**PACK_SCORING,
                   "roots": ("haenv/p3/score.py", "haenv/p3/gold.py"),
                   "judge_registry": "haenv/p3/plugin.py",
                   "optional_symbol_roots": (("haenv/p3/query.py", ("ADAPTER",)),),
                   "fields": ("p3_*",)},
    "P4_JUDGING": {**PACK_SCORING,
                   "roots": ("haenv/p4/score.py", "haenv/p4/gold.py"),
                   "judge_registry": "haenv/p4/plugin.py",
                   "optional_symbol_roots": (("haenv/p4/query.py", ("ADAPTER",)),),
                   "fields": ("p4_*",)},
    "M2_JUDGING": {**PACK_SCORING,
                   "roots": ("haenv/m2/score.py", "haenv/m2/hooks.py", "haenv/m2/round2.py", "haenv/m2/history.py"),
                   "judge_registry": "haenv/m2/plugin.py",
                   "optional_symbol_roots": (("haenv/m2/query.py", ("ADAPTER",)),
                                             ("haenv/interaction/query/gated.py", ("solve_gated_round",))),
                   "fields": ("m2_*",)},
    "PACK2_JUDGING": {**PACK_SCORING,
                      "roots": ("haenv/pack2/score.py",),
                      "judge_registry": "haenv/pack2/plugin.py",
                      "fields": ("pack2_*",)},
}
