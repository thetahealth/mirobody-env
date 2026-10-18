"""Compatibility imports; implementations live in their explicit owner modules.

Internal callers should import from the canonical owners; this module preserves
existing public entrypoints without forwarding mutable module globals.
"""

from haenv.scoring.analysis.ceilings import _noise_alias
from haenv.scoring.analysis.ceilings import batch_discrimination_gate
from haenv.scoring.analysis.ceilings import constant_ceilings
from haenv.scoring.analysis.ceilings import degenerate_ceiling_gate
from haenv.scoring.analysis.ceilings import degenerate_ceilings

from haenv.scoring.analysis.costs import budget_ceiling_by_solver
from haenv.scoring.analysis.costs import cost_efficiency_analysis

from haenv.scoring.analysis.decisions import INSUFFICIENT_KIND
from haenv.scoring.analysis.decisions import JOIN_CLASSES
from haenv.scoring.analysis.decisions import _CEILING_MAX_DIFF_ITEMS
from haenv.scoring.analysis.decisions import _scope_const_floor
from haenv.scoring.analysis.decisions import abstention_utility_of
from haenv.scoring.analysis.decisions import answer_space_diff
from haenv.scoring.analysis.decisions import chance_corrected
from haenv.scoring.analysis.decisions import join_class_of
from haenv.scoring.analysis.decisions import join_scored_classes
from haenv.scoring.analysis.decisions import review_macro_of

from haenv.scoring.analysis.dimensions import CORRECTNESS_METRICS
from haenv.scoring.analysis.dimensions import METRIC_KINDS
from haenv.scoring.analysis.dimensions import MIN_CELLS_FOR_EXERCISE
from haenv.scoring.analysis.dimensions import MIN_GRIDS_FOR_SCORE
from haenv.scoring.analysis.dimensions import MIN_ITEMS_FOR_EXERCISE
from haenv.scoring.analysis.dimensions import _QNAMES
from haenv.scoring.analysis.dimensions import _REC_KEY
from haenv.scoring.analysis.dimensions import _rowdim
from haenv.scoring.analysis.dimensions import dimension_health
from haenv.scoring.analysis.dimensions import real_solver_pool
from haenv.scoring.analysis.dimensions import scored_dims_still_healthy
from haenv.scoring.analysis.dimensions import unexercised_dims

from haenv.scoring.analysis.discrimination import item_discrimination

from haenv.scoring.analysis.gates import NON_HARM_GATES
from haenv.scoring.analysis.gates import PROPENSITY_GATES
from haenv.scoring.analysis.gates import gate_family_of
from haenv.scoring.analysis.gates import gate_multiplier
from haenv.scoring.analysis.gates import kernel_gate_kinds

from haenv.scoring.analysis.precision import _EXACT
from haenv.scoring.analysis.precision import _fsum_mean
from haenv.scoring.analysis.precision import _mean
from haenv.scoring.analysis.precision import _rd
from haenv.scoring.analysis.precision import exact_precision

from haenv.scoring.analysis.probes import noop_reading

from haenv.scoring.analysis.provenance import NotPublishable
from haenv.scoring.analysis.provenance import assert_publishable
from haenv.scoring.analysis.provenance import judging_provenance

from haenv.scoring.analysis.ranking import REPORTED_DIAGNOSTICS
from haenv.scoring.analysis.ranking import _apply_one_ruler
from haenv.scoring.analysis.ranking import _board_order
from haenv.scoring.analysis.ranking import rank_ddx
from haenv.scoring.analysis.ranking import rank_ddx_exact
from haenv.scoring.analysis.ranking import rank_models

from haenv.scoring.analysis.tracks import BLIND_BASELINES
from haenv.scoring.analysis.tracks import CORE_DIM_TO_REC
from haenv.scoring.analysis.tracks import HEURISTIC_BASELINES
from haenv.scoring.analysis.tracks import NA_TRACKS
from haenv.scoring.analysis.tracks import _CORE_NAMES
from haenv.scoring.analysis.tracks import _NO_HEADROOM_EPS
from haenv.scoring.analysis.tracks import _b_floor
from haenv.scoring.analysis.tracks import _c_floor
from haenv.scoring.analysis.tracks import _core_from_profile
from haenv.scoring.analysis.tracks import applicable_tracks
from haenv.scoring.analysis.tracks import is_ddx_batch
from haenv.scoring.analysis.tracks import noinfo_floors
from haenv.scoring.analysis.tracks import skill_over_floor
from haenv.scoring.analysis.tracks import stub_geometry_scope
from haenv.scoring.analysis.tracks import task_kind_of

from haenv.reporting.formatting import (
    _ceiling_line,
    _fmt,
    _render_stamp,
    board_rank_labels,
    floors_note,
    gate_cell,
    gate_state_note,
    geometry_scope_note,
)

from haenv.reporting.readouts import (
    disputed_gold_hits,
    noise_floor_for,
)
