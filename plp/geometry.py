"""Placement primitives for the 3D-PLP decoder.

Coordinates: container axes (x, y, z), units cm, half-open boxes
[b.x, b.x+b.w) x [b.y, b.y+b.d) x [b.z, b.z+b.h).

Hard constraints enforced at placement time (spec D4'):
  C1  no interior overlap
  C2  inside the container
  C3  bottom face >= tau (0.5) covered by floor (z=0) or top faces at level z
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Placed box: lower-back corner (x, y, z), extents (w, d, h)."""
    x: int
    y: int
    z: int
    w: int
    d: int
    h: int


def overlaps(a: Box, b: Box) -> bool:
    """C1 violation: strict interior volume overlap. Touching faces is not overlap."""
    return (a.x < b.x + b.w and b.x < a.x + a.w
            and a.y < b.y + b.d and b.y < a.y + a.d
            and a.z < b.z + b.h and b.z < a.z + a.h)


def in_container(b: Box, L: int, W: int, H: int) -> bool:
    """C2: box fully inside the container (touching the boundary is allowed)."""
    return (b.x >= 0 and b.y >= 0 and b.z >= 0
            and b.x + b.w <= L and b.y + b.d <= W and b.z + b.h <= H)


def support_ratio(placed, x: int, y: int, z: int, w: int, d: int) -> float:
    """C3: fraction of the bottom face (w x d at (x, y)) covered by support.

    Support = the container floor when z == 0, else the union of top faces of
    placed boxes whose top level is exactly z (a gap below the bottom face
    cannot carry load). Sum of overlaps; denominator is the bottom area.
    """
    if z == 0:
        return 1.0
    bottom = w * d
    total = 0
    for j in placed:
        if j.z + j.h != z:
            continue
        ox = min(x + w, j.x + j.w) - max(x, j.x)
        if ox <= 0:
            continue
        oy = min(y + d, j.y + j.d) - max(y, j.y)
        if oy > 0:
            total += ox * oy
    return total / bottom
