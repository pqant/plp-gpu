"""Parser for BR-family 3D container-loading instances.

File format (OR-Library thpack / kcliu2 CLP-Datasets "original" format):

    line 0:     P                   number of instances in the file
    per instance:
        p [seed]                    problem id (seed optional)
        L W H                       container dimensions (cm)
        n                           number of box types
        i l lf w wf h hf qty        one line per box type (i = 1..n)

lf/wf/hf are 0/1: whether the box may stand on that axis, i.e. whether that
dimension may become the vertical one in a placement.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


class InstanceFormatError(ValueError):
    """Instance file violates the BR/thpack format."""


@dataclass(frozen=True)
class Item:
    length: int
    length_can_stand: bool
    width: int
    width_can_stand: bool
    height: int
    height_can_stand: bool
    qty: int

    @property
    def volume(self) -> int:
        return self.length * self.width * self.height


@dataclass(frozen=True)
class Instance:
    container_length: int
    container_width: int
    container_height: int
    items: tuple

    @property
    def container_volume(self) -> int:
        return self.container_length * self.container_width * self.container_height

    @property
    def total_qty(self) -> int:
        return sum(i.qty for i in self.items)

    @property
    def total_item_volume(self) -> int:
        return sum(i.volume * i.qty for i in self.items)


def _int(token: str, what: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise InstanceFormatError(f"not an integer: {token!r} ({what})") from None


def parse(text: str) -> list[Instance]:
    """Parse one or more BR/thpack instances. Raises InstanceFormatError on any violation."""
    tokens = [line.split() for line in text.splitlines() if line.strip()]
    if not tokens:
        raise InstanceFormatError("empty file")
    try:
        P = int(tokens[0][0])
    except ValueError:
        raise InstanceFormatError(f"line 0 must be an integer count, got {tokens[0][0]!r}") from None
    if P <= 0:
        raise InstanceFormatError(f"instance count must be >= 1, got {P}")

    pos = 1
    instances: list[Instance] = []
    for _ in range(P):
        # problem line: "p" or "p seed"
        if pos >= len(tokens) or not 1 <= len(tokens[pos]) <= 2:
            got = tokens[pos] if pos < len(tokens) else "<EOF>"
            raise InstanceFormatError(f"malformed problem line: {got}")
        pos += 1
        # container line: L W H
        if pos >= len(tokens) or len(tokens[pos]) != 3:
            raise InstanceFormatError("container line must have 3 dimensions")
        L, W, H = (_int(t, "container dimension") for t in tokens[pos])
        if min(L, W, H) <= 0:
            raise InstanceFormatError(f"container dimensions must be positive: {tokens[pos]}")
        pos += 1
        # type-count line: n
        if pos >= len(tokens) or len(tokens[pos]) != 1:
            raise InstanceFormatError("type-count line must be a single integer")
        n = _int(tokens[pos][0], "type count")
        pos += 1
        # item lines: i l lf w wf h hf qty
        items = []
        for k in range(1, n + 1):
            if pos >= len(tokens):
                raise InstanceFormatError(f"instance truncated: expected {n} item lines, got {k - 1}")
            t = tokens[pos]
            pos += 1
            if len(t) != 8:
                raise InstanceFormatError(f"item line must have 8 tokens, got {len(t)}: {t}")
            idx = _int(t[0], "item index")
            if idx != k:
                raise InstanceFormatError(f"item index must be sequential, expected {k}, got {idx}")
            dims = [_int(x, "item dimension") for x in (t[1], t[3], t[5])]
            if any(d <= 0 for d in dims):
                raise InstanceFormatError(f"item dimensions must be positive: {t}")
            flags = [t[2], t[4], t[6]]
            if any(f not in ("0", "1") for f in flags):
                raise InstanceFormatError(f"orientation flags must be 0/1: {t}")
            qty = _int(t[7], "item qty")
            if qty < 0:
                raise InstanceFormatError(f"item qty must be >= 0: {t}")
            items.append(Item(dims[0], flags[0] == "1", dims[1], flags[1] == "1",
                              dims[2], flags[2] == "1", qty))
        instances.append(Instance(L, W, H, tuple(items)))

    if pos != len(tokens):
        raise InstanceFormatError(f"trailing data after {P} instances: {tokens[pos]}")
    return instances


def load(path: str) -> list[Instance]:
    with open(path) as f:
        return parse(f.read())


def load_instance(path: str) -> Instance:
    """Load a single-instance data file (P == 1)."""
    insts = load(path)
    if len(insts) != 1:
        raise InstanceFormatError(f"expected exactly 1 instance in {path}, got {len(insts)}")
    return insts[0]


def allowed_orientations(item: Item) -> list[tuple[int, int, int]]:
    """All (w, d, h) placements for one box along container axes.

    A dimension may be the vertical one only if its stand flag is 1;
    each permitted vertical axis yields 2 rotations. Max 6.
    """
    out = []
    l, w, h = item.length, item.width, item.height
    if item.length_can_stand:
        out += [(w, h, l), (h, w, l)]
    if item.width_can_stand:
        out += [(l, h, w), (h, l, w)]
    if item.height_can_stand:
        out += [(l, w, h), (w, l, h)]
    return out
