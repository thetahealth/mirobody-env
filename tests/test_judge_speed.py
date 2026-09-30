"""Token reduction must be reversible; concurrency must be bounded and resumable."""
import json
import threading

import pytest

from haenv.judge_compaction import pack_reference, unpack_reference
from haenv.judge_scheduler import run_bounded
from haenv.semantic_parallel import MAX_JUDGE_LANES


def reference():
    return {"diagnosis": "A", "visible_case": {
        "evidence_ledger": [{"text": "Do not drop this clinical qualifier", "ts": 3}],
        "longitudinal_data": {"steps": [{"ts": i, "value": 5.5, "flag": None} for i in range(100)],
                              "mixed": [{"ts": 1, "value": 4}, {"ts": 2, "value": None, "note": "missing"}]}}}


def test_reference_compression_is_lossless_and_nonmutating():
    original = reference(); before = json.dumps(original)
    compact = pack_reference(original)
    assert unpack_reference(compact) == original
    assert json.dumps(original) == before
    assert len(json.dumps(compact)) < len(before) * .7
    assert compact["visible_case"]["evidence_ledger"] == original["visible_case"]["evidence_ledger"]
    assert compact["visible_case"]["longitudinal_data"]["mixed"] == original["visible_case"]["longitudinal_data"]["mixed"]


def test_roundtrip_preserves_time_order_types_and_nulls():
    data = reference()
    data["visible_case"]["longitudinal_data"]["steps"] = [
        {"ts": 9, "value": "positive", "flag": False},
        {"ts": 2, "value": None, "flag": True}]
    assert unpack_reference(pack_reference(data)) == data


def test_four_workers_really_overlap_without_exceeding_cap():
    lock = threading.Lock(); barrier = threading.Barrier(4)
    active = 0; peak = 0
    def work(item):
        nonlocal active, peak
        with lock:active += 1;peak = max(peak, active)
        barrier.wait(timeout=3)
        with lock:active -= 1
        return item * 2
    assert run_bounded(list(range(4)), work, concurrency=4) == [0, 2, 4, 6]
    assert peak == 4


def test_stop_prevents_queued_work_but_keeps_finished_result():
    stop = threading.Event();calls = []
    def work(item):calls.append(item);stop.set();return "done"
    assert run_bounded([1, 2, 3], work, concurrency=1, stop=stop) == ["done", None, None]
    assert calls == [1]


@pytest.mark.parametrize("n", [0, -1, MAX_JUDGE_LANES + 1, True])
def test_invalid_concurrency_rejected_before_dispatch(n):
    with pytest.raises(ValueError):
        run_bounded([1], lambda _:pytest.fail("dispatched"), concurrency=n)


def test_scheduler_and_cell_runner_share_one_lane_ceiling():
    import haenv.judge_scheduler as scheduler, haenv.semantic_parallel as parallel
    assert scheduler.MAX_JUDGE_LANES is parallel.MAX_JUDGE_LANES == MAX_JUDGE_LANES == 64
    assert run_bounded([], lambda x: x, concurrency=MAX_JUDGE_LANES) == []


def test_max_lanes_really_run_at_once():
    # A barrier of MAX_JUDGE_LANES parties only opens if that many workers are live together;
    # a pool smaller than the lane count would time out.
    barrier = threading.Barrier(MAX_JUDGE_LANES)
    def work(item):
        barrier.wait(timeout=20)
        return item
    items = list(range(MAX_JUDGE_LANES))
    assert run_bounded(items, work, concurrency=MAX_JUDGE_LANES) == items


def test_live_prompt_uses_lossless_tables_and_shared_reference_prefix():
    from haenv.semantic_inputs import anonymous_prompt
    ref = reference()
    first, body = anonymous_prompt({"dx": "first rule"}, ref, {"differential": [{"diagnosis": "A"}]})
    second, _ = anonymous_prompt({"dx": "second rule"}, ref, {"differential": [{"diagnosis": "B"}]})
    assert first.index("REFERENCE:") < first.index("ATOMIC CRITERIA:")
    assert first.split("ATOMIC CRITERIA:")[0] == second.split("ATOMIC CRITERIA:")[0]
    saved = json.loads(first.split("REFERENCE:\n",1)[1].split("\n\nATOMIC CRITERIA:",1)[0])
    assert unpack_reference(saved) == ref
    assert json.loads(body) == {"differential": [{"diagnosis": "A"}]}
