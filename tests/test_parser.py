"""Tests for plp.parser — BR-family instance parser.

Run from experiments/plp/:
    packages/paper-forge/.venv/bin/python -m pytest tests/ -v
"""
import json
import os

import pytest

from plp.parser import (
    Instance,
    InstanceFormatError,
    Item,
    allowed_orientations,
    load_instance,
    parse,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def item(l, lf, w, wf, h, hf, qty):
    return Item(length=l, length_can_stand=lf, width=w, width_can_stand=wf,
                height=h, height_can_stand=hf, qty=qty)


# ---------------------------------------------------------------- real data

def test_parse_real_br1_050():
    inst = load_instance(os.path.join(DATA_DIR, "instances", "BR1_050.txt"))
    assert (inst.container_length, inst.container_width, inst.container_height) == (587, 233, 220)
    assert len(inst.items) == 3
    assert inst.items[0] == item(60, True, 40, True, 32, True, 64)
    assert inst.items[1] == item(98, True, 75, True, 55, True, 40)
    assert inst.items[2] == item(60, True, 59, True, 39, True, 64)
    assert inst.total_qty == 168
    assert inst.container_volume == 587 * 233 * 220


def test_all_15_instances_match_manifest():
    with open(os.path.join(DATA_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    entries = manifest["instances"]
    assert len(entries) == 15
    for e in entries:
        path = os.path.join(DATA_DIR, e["file"])
        assert os.path.exists(path), path
        inst = load_instance(path)
        assert (inst.container_length, inst.container_width, inst.container_height) == tuple(e["container_cm"]), e["name"]
        assert len(inst.items) == e["n_types"], e["name"]
        assert inst.total_qty == e["total_qty"], e["name"]
        assert inst.container_volume == e["container_volume_cm3"], e["name"]
        assert inst.total_item_volume == e["item_volume_cm3"], e["name"]


# ------------------------------------------------------------------ fixtures

def two_instance_text():
    return "\n".join([
        "2",
        # instance 1
        "1 1001",
        "587 233 220",
        "2",
        "1 60 1 40 1 32 1 64",
        "2 98 1 75 1 55 1 40",
        # instance 2 (different container, different types)
        "2 2002",
        "1200 233 259",
        "1",
        "1 100 0 50 1 40 1 30",
    ])


def one_instance_text(body_lines):
    # body_lines: [container_line, n_line, *item_lines]; problem line fixed
    return "\n".join(["1", "1 1001", *body_lines])


def test_parse_handles_multiple_instances():
    insts = parse(two_instance_text())
    assert len(insts) == 2
    assert (insts[0].container_length, insts[0].container_width, insts[0].container_height) == (587, 233, 220)
    assert len(insts[0].items) == 2
    assert (insts[1].container_length, insts[1].container_width, insts[1].container_height) == (1200, 233, 259)
    assert insts[1].items[0] == item(100, False, 50, True, 40, True, 30)


def test_item_line_field_mapping():
    # token order: i, l, l_flag, w, w_flag, h, h_flag, qty
    insts = parse(one_instance_text(["587 233 220", "1", "1 10 0 20 1 30 0 5"]))
    it = insts[0].items[0]
    assert it == item(10, False, 20, True, 30, False, 5)
    assert it.volume == 10 * 20 * 30


# ------------------------------------------------- orientation derivation

def test_allowed_orientations_all_flags():
    # orientation = (w, d, h) along container axes; vertical axis must be permitted
    it = item(2, True, 3, True, 4, True, 1)
    assert set(allowed_orientations(it)) == {
        (3, 4, 2), (4, 3, 2),   # length vertical
        (2, 4, 3), (4, 2, 3),   # width vertical
        (2, 3, 4), (3, 2, 4),   # height vertical
    }


def test_allowed_orientations_partial_flags():
    it = item(2, True, 3, False, 4, False, 1)
    assert set(allowed_orientations(it)) == {(3, 4, 2), (4, 3, 2)}


def test_allowed_orientations_no_flags():
    it = item(2, False, 3, False, 4, False, 1)
    assert allowed_orientations(it) == []


# ------------------------------------------------------------------- errors

@pytest.mark.parametrize("bad_text, why", [
    (one_instance_text(["587 233 220", "1", "1 10 1 20 1 30 1"]), "item line with 7 tokens"),
    (one_instance_text(["587 233 220", "1", "1 10 2 20 1 30 1 5"]), "flag value 2"),
    (one_instance_text(["587 0 220", "1", "1 10 1 20 1 30 1 5"]), "zero container dimension"),
    (one_instance_text(["587 233 220", "1", "1 0 1 20 1 30 1 5"]), "zero item dimension"),
    (one_instance_text(["587 233 220", "1", "1 10 1 20 1 30 1 -5"]), "negative qty"),
    (one_instance_text(["587 233 220", "1", "1 10.5 1 20 1 30 1 5"]), "non-integer dimension"),
    (one_instance_text(["587 233 220", "2", "1 10 1 20 1 30 1 5", "3 5 1 6 1 7 1 2"]), "non-sequential item index"),
])
def test_rejects_malformed_instance(bad_text, why):
    with pytest.raises(InstanceFormatError):
        parse(bad_text)


def test_rejects_declared_count_mismatch():
    # declares P=2 but provides only one instance block
    text = "\n".join(["2", "1 1001", "587 233 220", "1", "1 10 1 20 1 30 1 5"])
    with pytest.raises(InstanceFormatError):
        parse(text)


def test_rejects_trailing_garbage():
    text = two_instance_text() + "\n999"
    with pytest.raises(InstanceFormatError):
        parse(text)


def test_rejects_zero_instance_count():
    with pytest.raises(InstanceFormatError):
        parse("0")


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_instance(os.path.join(DATA_DIR, "instances", "NOPE_000.txt"))
