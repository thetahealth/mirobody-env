"""Minimal end-to-end demo of a world-side plugin, installed as a separate package.

Wiring: `world()` calls the `world_plugins.register_*` functions; pyproject.toml
exposes it under the entry-point group `haenv.world.demo`; a job loads it with
`plugins: {world: ["haenv.world.demo"]}`. The plugin enters `world_sha`.

A world-side plugin cannot write label-bearing streams such as `weight`
(`_forbidden_demo` shows the error), and registered streams still pass the
generation gates.

SYNTHETIC, for evaluation only, not medical advice.
"""
from __future__ import annotations

SOURCE = "haenv.world.demo"


def _demo_injector(points, *, rng_key: str, drop_frac: float = 0.0, **_):
    """Demo injector: drops the trailing `drop_frac` of the points."""
    if not points or drop_frac <= 0:
        return points
    keep = max(1, int(len(points) * (1.0 - drop_frac)))
    return list(points[:keep])


def world() -> None:
    """The entry point points here. Calls the registration points internally, returns nothing."""
    from haenv import world_plugins as wp

    # 1. A physiological stream block beyond the ones built in. It passes the same field checks
    #    as `registry/physio_streams.yaml` (group, loading, idio_sd, bounds, transform, source,
    #    review); no device in the demo case emits it, so it changes no output by itself.
    wp.register_stream(
        "demo_resting_hr",
        {"group": "cardio", "loading": 0.0, "idio_sd": 0.1, "lo": 30.0, "hi": 140.0,
         "max_step": 20.0, "transform": "identity",
         "source": "SYNTHETIC 演示流,不代表任何真实设备的测量特性", "review": "pending"},
        source=SOURCE)

    # 2. A life event that takes effect in the demo case. It is a full pool item (`age_w`
    #    and a `cond` declaration for every openable axis are required, the pool checks refuse
    #    it otherwise); the age prior is set far above the in-repo items (about 1) so that the
    #    demo case draws it (its density asks for one life event), which makes `demo` differ from `demo-noplugin` in the payload.
    wp.register_event_pool(
        "demo_new_pillow",
        {"text": "换了新枕头,头两晚睡得浅", "context": "居家用品更换",
         "topic": "demo_new_pillow", "tags": ["sleep"], "exertion": False,
         "age_w": [1000.0, 1000.0, 1000.0],
         "cond": {"season": {"w": {"spring": 1.0, "summer": 1.0, "autumn": 1.0, "winter": 1.0}},
                  "activity": {"w": {"sedentary": 1.0, "active": 1.0}}}},
        source=SOURCE, kind="life_event")

    # 3. A treatment effect (does not touch weight). It passes the same field checks as the
    #    in-repo table (`drug_effects._doc`); `null` is allowed where the table allows it.
    wp.register_effect(
        "demo_drug",
        {"total_hba1c_pp": -0.5, "trial_weight_kg": -1.0, "total_fpg_mmol": None,
         "sd_change_hba1c_pp": None, "readout_weeks": 26,
         "background": "monotherapy_vs_placebo", "estimand": "efficacy",
         "cohorts": ["T2D"], "source": "SYNTHETIC 演示值,无文献出处", "review": "pending"},
        source=SOURCE)

    # 4. A post-processing injector; `streams` is required, or it acts on no stream.
    wp.register_post_injector(
        "demo_tail_drop", _demo_injector,
        params={"streams": ["demo_resting_hr"], "drop_frac": 0.05,
                "calibration": "by_design",
                "why": "演示用;真实缺失率见 registry/artifact_rates.yaml"},
        source=SOURCE)


def _forbidden_demo() -> None:
    """Registers `weight`, which raises `WorldPluginError`; not called by `world()`."""
    from haenv import world_plugins as wp
    wp.register_stream("weight", {"unit": "kg"}, source=SOURCE)
