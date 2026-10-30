"""Tests for plp.solver — CPU baseline decoder.

Covers: orientation dedup, extreme-point candidate set, D4'/D5 FSR + stability
(hand-computed), D4' acceptance (theta fallback), decode pass (C1/C2/C3,
volume-desc vs input order, first-fit == eps=0), run() (single/multi pass,
fallback flags), 15 real-instance smoke, CLI.

Run from experiments/plp/:
    packages/paper-forge/.venv/bin/python -m pytest tests/ -v
"""
import json
import os
import random
import subprocess
import sys

import pytest

from plp.geometry import in_container, overlaps, support_ratio
from plp.parser import Item, Instance, load_instance
from plp.solver import (
    TAU,
    THETA,
    PlacedBox,
    candidate_points,
    compute_fsr,
    decode_pass,
    is_accepted,
    mass_kg,
    orientations,
    run,
    select_best,
    stability_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PLP_DIR = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(PLP_DIR, "data")


def pbox(x, y, z, w, d, h, t=0):
    return PlacedBox(x=x, y=y, z=z, w=w, d=d, h=h, item_type=t)


def item(l, w, h, qty):
    return Item(l, True, w, True, h, True, qty)


def packing_of(placed):
    from plp.solver import Packing
    return Packing(tuple(placed), len(placed),
                   sum(b.w * b.d * b.h for b in placed),
                   stability_score(placed))


# ----------------------------------------------------- orientation dedup (D1)

def test_orientations_cube_single_variant():
    assert orientations(item(10, 10, 10, 1)) == [(10, 10, 10)]


def test_orientations_preserves_parser_order():
    it = Item(2, True, 3, True, 4, True, 1)
    assert orientations(it) == [
        (3, 4, 2), (4, 3, 2),   # length vertical
        (2, 4, 3), (4, 2, 3),   # width vertical
        (2, 3, 4), (3, 2, 4),   # height vertical
    ]


# ------------------------------------------------- extreme-point set (D1)

def test_candidates_initial_origin_only():
    assert candidate_points([]) == {(0, 0, 0)}


def test_candidates_after_one_box_eight_corners():
    placed = [pbox(0, 0, 0, 10, 10, 10)]
    pts = candidate_points(placed)
    assert pts == {(x, y, z) for x in (0, 10) for y in (0, 10) for z in (0, 10)}


def test_candidates_drop_points_strictly_inside_a_box():
    a = pbox(0, 0, 0, 10, 10, 10)
    b = pbox(5, 5, 5, 10, 10, 10)   # physically overlapping: rule test only
    pts = candidate_points([a, b])
    assert (10, 10, 10) not in pts          # a's corner, strictly inside b
    assert (5, 5, 5) not in pts             # b's corner, strictly inside a
    assert len(pts) == 14                   # 8 + 8 corners - 2 dropped


def test_candidates_boundary_points_kept():
    # a corner on a box's face (not strictly inside) stays a candidate
    placed = [pbox(0, 0, 0, 10, 10, 10)]
    assert (10, 0, 0) in candidate_points(placed)


# ---------------------------------------------------------------- mass model

def test_mass_kg_from_cm3():
    # rho = 50 kg/m3; 10x10x10 cm = 1e-3 m3 -> 0.05 kg
    assert mass_kg(1000) == pytest.approx(0.05)


# ------------------------------------------- FSR / stability (D4', D5, D6)

def test_fsr_single_box_on_floor():
    # no load above: FSR = mu/a_lat = 0.2/0.3 < 1 -> lone box is unstable
    placed = [pbox(0, 0, 0, 10, 10, 10)]
    assert compute_fsr(placed)[0] == pytest.approx(2 / 3)
    assert stability_score(placed) == 0.0


def test_fsr_two_box_stack_hand_calc():
    # hand calc: FSR = 0.667 * (1 + L/m). Top: L=0 -> 2/3. Bottom: L=m -> 4/3.
    placed = [pbox(0, 0, 0, 10, 10, 10), pbox(0, 0, 10, 10, 10, 10)]
    fsr = compute_fsr(placed)
    assert fsr[0] == pytest.approx(4 / 3)
    assert fsr[1] == pytest.approx(2 / 3)
    assert stability_score(placed) == 0.5


def test_fsr_partial_support_distribution():
    # Equal masses (14x10x10). C's bottom [5,19) x [0,10) contacts A's top
    # [0,14) over 9 and B's top [14,28) over 5 -> load split 9/14 and 5/14:
    #   FSR_A = (2/3)(1 + 9/14) = 23/21 >= 1, FSR_B = (2/3)(1 + 5/14) = 19/21 < 1
    a = pbox(0, 0, 0, 14, 10, 10)
    b = pbox(14, 0, 0, 14, 10, 10)
    c = pbox(5, 0, 10, 14, 10, 10)
    fsr = compute_fsr([a, b, c])
    assert fsr[0] == pytest.approx(23 / 21)
    assert fsr[1] == pytest.approx(19 / 21)
    assert fsr[2] == pytest.approx(2 / 3)
    assert stability_score([a, b, c]) == pytest.approx(1 / 3)


def test_fsr_transitive_three_high_tower():
    # tower: bottom carries 3m, middle 2m, top m -> FSR 2, 4/3, 2/3
    placed = [pbox(0, 0, 0, 10, 10, 10), pbox(0, 0, 10, 10, 10, 10),
              pbox(0, 0, 20, 10, 10, 10)]
    fsr = compute_fsr(placed)
    assert fsr[0] == pytest.approx(2.0)
    assert fsr[1] == pytest.approx(4 / 3)
    assert fsr[2] == pytest.approx(2 / 3)
    assert stability_score(placed) == pytest.approx(2 / 3)


# ------------------------------------------------- D4' acceptance (theta)

def test_is_accepted_inclusive_and_theta_none():
    assert is_accepted(0.9, 0.9) is True
    assert is_accepted(0.8999, 0.9) is False
    assert is_accepted(0.0, None) is True    # V2: no filter


def test_select_best_fallback_when_none_accepted():
    # both packings are single boxes on the floor -> score 0 < theta
    packings = [
        packing_of([pbox(0, 0, 0, 10, 10, 10)]),                      # vol 1000
        packing_of([pbox(0, 0, 0, 30, 10, 10)]),                      # vol 3000
    ]
    best, fallback = select_best(packings, theta=0.9)
    assert fallback is True
    assert best.volume == 3000      # overall best volume, honest flag


def test_select_best_highest_volume_among_accepted():
    rejected = [packing_of([pbox(0, 0, 0, 30, 10, 10)])]             # score 0
    tower10 = [pbox(0, 0, 10 * i, 10, 10, 10) for i in range(10)]    # score 9/10 = 0.9
    tower11 = [pbox(0, 0, 10 * i, 10, 10, 10) for i in range(11)]    # score 10/11
    # tower10 sits exactly on the theta boundary -> accepted (inclusive)
    best, fallback = select_best(rejected + [packing_of(tower10)], theta=0.9)
    assert fallback is False
    assert best.volume == 10000
    # among accepted, higher volume wins
    best2, _ = select_best(rejected + [packing_of(tower10), packing_of(tower11)], theta=0.9)
    assert best2.volume == 11000


# ---------------------------------------------------------- decode pass (D2/D3)

def test_pass_two_boxes_side_by_side_bottom_left_back():
    inst = Instance(100, 100, 100, (item(60, 50, 50, 2),))
    placed = decode_pass(inst, eps=0.0)
    assert len(placed) == 2
    # first orientation of item(60,50,50): length vertical -> (w,d,h)=(50,50,60)
    assert (placed[0].x, placed[0].y, placed[0].z) == (0, 0, 0)
    assert (placed[0].w, placed[0].d, placed[0].h) == (50, 50, 60)
    # greedy (z, y, x) per spec D3: same level, then smallest y, then x
    # -> (50,0,0) (y=0) precedes (0,50,0) (y=50); both feasible here
    assert (placed[1].x, placed[1].y, placed[1].z) == (50, 0, 0)
    assert not overlaps(placed[0], placed[1])


def test_pass_stacks_when_floor_is_full():
    inst = Instance(10, 10, 30, (item(10, 10, 10, 3),))
    placed = decode_pass(inst, eps=0.0)
    assert len(placed) == 3
    assert [b.z for b in placed] == [0, 10, 20]
    assert all(in_container(b, 10, 10, 30) for b in placed)


def test_pass_skips_unplaceable_box_and_continues():
    # A 15^3 (vol 3375) fits; B 25x10x10 (vol 2500) exceeds the 20 cm box in
    # every orientation. Volume-desc order tries A first, then B must be
    # skipped without breaking the pass.
    inst = Instance(20, 20, 20, (item(15, 15, 15, 1), item(25, 10, 10, 1)))
    placed = decode_pass(inst, eps=0.0)
    assert len(placed) == 1
    assert placed[0].item_type == 0


def test_volume_desc_vs_input_order():
    # type 0 small (10^3), type 1 large (50^3)
    inst = Instance(100, 100, 100, (item(10, 10, 10, 1), item(50, 50, 50, 1)))
    vol = decode_pass(inst, order="volume", eps=0.0)
    inp = decode_pass(inst, order="input", eps=0.0)
    assert vol[0].item_type == 1      # volume-desc: large first
    assert inp[0].item_type == 0      # input order: type 0 first


def test_first_fit_identical_to_eps_zero():
    inst = Instance(100, 100, 100, (item(60, 50, 50, 2), item(30, 30, 30, 4)))
    assert decode_pass(inst, first_fit=True) == decode_pass(inst, eps=0.0)


def test_same_seed_same_result_with_epsilon():
    inst = Instance(100, 100, 100, (item(60, 50, 50, 2), item(30, 30, 30, 4)))
    a = decode_pass(inst, eps=0.3, rng=random.Random(42))
    b = decode_pass(inst, eps=0.3, rng=random.Random(42))
    assert a == b


# ----------------------------------------------------------------------- run

def test_run_single_pass_fallback_flag():
    # single box on floor -> score 0 < theta -> fallback to overall best
    inst = Instance(100, 100, 100, (item(10, 10, 10, 1),))
    r = run(inst, seed=1, time_limit_s=5, single_pass=True)
    assert r.n_passes == 1
    assert r.volume == 1000
    assert r.fallback is True
    assert r.score == pytest.approx(0.0)


def test_run_theta_none_never_fallback():
    inst = Instance(100, 100, 100, (item(10, 10, 10, 1),))
    r = run(inst, seed=1, time_limit_s=5, single_pass=True, theta=None)
    assert r.fallback is False
    assert r.volume == 1000


def test_run_tower10_accepted_end_to_end():
    # 10 stacked 10^3 boxes: score (n-1)/n = 0.9 >= theta -> accepted, no fallback
    inst = Instance(10, 10, 100, (item(10, 10, 10, 10),))
    r = run(inst, seed=0, time_limit_s=5, single_pass=True)
    assert r.n_boxes == 10
    assert r.volume == 10000
    assert r.score == pytest.approx(0.9)
    assert r.fallback is False


def test_run_tower9_fallback_end_to_end():
    # 9 stacked: score 8/9 < 0.9 -> fallback (honest flag), volume still reported
    inst = Instance(10, 10, 90, (item(10, 10, 10, 9),))
    r = run(inst, seed=0, time_limit_s=5, single_pass=True)
    assert r.n_boxes == 9
    assert r.score == pytest.approx(8 / 9)
    assert r.fallback is True
    assert r.volume == 9000


def test_run_multi_pass_keeps_best_and_respects_seed():
    inst = Instance(60, 60, 60, (item(15, 15, 15, 16), item(12, 12, 12, 12)))
    single = run(inst, seed=7, time_limit_s=5, single_pass=True)
    multi = run(inst, seed=7, time_limit_s=2.0, eps=0.3)
    assert multi.n_passes >= 2
    assert multi.volume >= single.volume     # best over passes >= first pass


# ------------------------------------------------- real instances (smoke)

def _manifest():
    with open(os.path.join(DATA_DIR, "manifest.json")) as f:
        return json.load(f)["instances"]


def test_all_15_instances_single_pass_valid():
    for e in _manifest():
        inst = load_instance(os.path.join(DATA_DIR, e["file"]))
        placed = decode_pass(inst, eps=0.0)
        assert len(placed) > 0, e["name"]
        L, W, H = inst.container_length, inst.container_width, inst.container_height
        for i in range(len(placed)):
            assert in_container(placed[i], L, W, H), e["name"]
            for j in range(i + 1, len(placed)):
                assert not overlaps(placed[i], placed[j]), (e["name"], i, j)
        for b in placed:   # final-state C3 recheck (support only ever grows)
            assert support_ratio(placed, b.x, b.y, b.z, b.w, b.d) >= TAU - 1e-12, e["name"]
        assert sum(b.w * b.d * b.h for b in placed) <= e["item_volume_cm3"]


# ----------------------------------------------------------------------- CLI

def test_cli_json_output():
    inst_file = os.path.join(DATA_DIR, "instances", "BR1_050.txt")
    out = subprocess.run(
        [sys.executable, "-m", "plp.solver", inst_file,
         "--single-pass", "--eps", "0", "--no-theta", "--json"],
        cwd=PLP_DIR, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["n_boxes"] > 0
    assert data["volume_cm3"] > 0
    assert data["time_s"] >= 0
    assert len(data["placements"]) == data["n_boxes"]
    assert len(data["placements"][0]) == 7   # item_type, x, y, z, w, d, h
