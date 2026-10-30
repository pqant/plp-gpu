"""Tests for plp.geometry — C1/C2/C3 placement primitives.

Run from experiments/plp/:
    packages/paper-forge/.venv/bin/python -m pytest tests/ -v
"""
from plp.geometry import Box, in_container, overlaps, support_ratio


def box(x, y, z, w, d, h):
    return Box(x=x, y=y, z=z, w=w, d=d, h=h)


# ------------------------------------------------------------- C1: no overlap

def test_overlap_interior_true():
    assert overlaps(box(0, 0, 0, 10, 10, 10), box(5, 0, 0, 10, 10, 10)) is True


def test_overlap_touching_faces_false():
    # sharing a face (x: [0,10) vs [10,20)) is NOT an overlap
    assert overlaps(box(0, 0, 0, 10, 10, 10), box(10, 0, 0, 10, 10, 10)) is False


def test_overlap_disjoint_false():
    assert overlaps(box(0, 0, 0, 10, 10, 10), box(20, 20, 20, 5, 5, 5)) is False


def test_overlap_stacked_z_touching_false():
    # top face of lower == bottom of upper: touching, not overlap
    assert overlaps(box(0, 0, 0, 10, 10, 10), box(0, 0, 10, 10, 10, 10)) is False


def test_overlap_different_heights_z_touching_false():
    # regression: lower top (98+56=154) touches upper bottom (z=154), but the
    # heights differ (56 vs 60) — the exact case that hides an a.h/b.h mixup
    assert overlaps(box(243, 95, 98, 81, 40, 56), box(243, 95, 154, 59, 39, 60)) is False


# ------------------------------------------------------------ C2: boundary

def test_in_container_exactly_filling_is_inside():
    assert in_container(box(0, 0, 0, 100, 50, 10), 100, 50, 10) is True


def test_in_container_one_cm_over_is_outside():
    assert in_container(box(0, 0, 0, 101, 50, 10), 100, 50, 10) is False
    assert in_container(box(0, 0, 0, 100, 50, 11), 100, 50, 10) is False


# ------------------------------------------------------------- C3: support

def test_support_floor_is_full():
    assert support_ratio([], 5, 5, 0, 10, 10) == 1.0


def test_support_full_on_box_top():
    lower = [box(0, 0, 0, 10, 10, 10)]
    assert support_ratio(lower, 0, 0, 10, 10, 10) == 1.0


def test_support_exactly_half_ratio():
    # bottom 20 wide starting at x=0: overlap with lower's top [0,10) is 10 of 20
    lower = [box(0, 0, 0, 10, 10, 10)]
    assert support_ratio(lower, 0, 0, 10, 20, 10) == 0.5


def test_support_below_half_ratio():
    # 3 of 10 cm of the bottom face on the box top -> 0.3
    lower = [box(0, 0, 0, 10, 10, 10)]
    assert support_ratio(lower, 7, 0, 10, 10, 10) == 0.3


def test_support_sums_over_multiple_boxes():
    a = box(0, 0, 0, 10, 10, 10)
    b = box(10, 0, 0, 10, 10, 10)
    # bottom 14 wide at x=5: 5 on a's top + 9 on b's top -> fully covered
    assert support_ratio([a, b], 5, 0, 10, 14, 10) == 1.0
    # bottom 10x10 at (0,5): only half the y-extent over a -> 0.5
    assert support_ratio([a, b], 0, 5, 10, 10, 10) == 0.5


def test_support_ignores_other_levels():
    # top face at z=5 cannot support a bottom at z=10
    lower = [box(0, 0, 0, 10, 10, 5)]
    assert support_ratio(lower, 0, 0, 10, 10, 10) == 0.0
