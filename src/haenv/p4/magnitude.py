"""A visible-header-only amplitude probe, independent of clinical evidence."""
from __future__ import annotations
import math


def readout(items):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    values, labels = [], []
    # Fold assignment must describe the same patients in every audit view.
    # Batch order and publication order carry no clinical information.
    for it in sorted(items, key=lambda r: r["case_id"]):
        b = it["sp"]["prediction_context"]["followup_p4"]
        from haenv.p4.records import clinical_view
        b = clinical_view(b)
        rows = {int(r["day"]): r for r in b["report"]}
        # Exactly the two printed header values: no dates, methods or record content.
        values.append(abs(math.log(float(rows[b["today"]]["value"]) / float(rows[b["previous_draw_day"]]["value"]))))
        labels.append(it["case"]["adjudication"]["followup_p4"]["class"])
    classes = sorted(set(labels))
    if len(classes) < 2 or min(labels.count(k) for k in classes) < 5:
        raise ValueError("magnitude probe needs five items of every class")
    X, y = np.asarray(values).reshape(-1, 1), np.asarray(labels)
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=1, random_state=20261012), X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=20261012), n_jobs=1)
    ba = float(balanced_accuracy_score(y, pred))
    ranges = {k: [min(v for v, c in zip(values, labels) if c == k), max(v for v, c in zip(values, labels) if c == k)] for k in classes}
    noise = ranges.get("analytic_biological_noise")
    overlap = (all(max(noise[0], span[0]) <= min(noise[1], span[1]) for k, span in ranges.items() if k != "analytic_biological_noise")
               if noise is not None else None)
    return {"n": len(labels), "balanced_accuracy": ba, "chance_corrected": (len(classes) * ba - 1) / (len(classes) - 1),
            "ranges_log": ranges, "overlap": overlap}
