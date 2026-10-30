"""CPU baseline decoder for the 3D-PLP solver.

Canonical implementation of the decoder rules (D1-D9 + D4'):

  D1  candidate positions = extreme points: origin + 8 corners of every
      placed box, minus points strictly inside a placed box.
      (No coordinate dominance: it can drop feasible positions, and the
      (z, y, x) greedy does not need it.)
  D2  item order: volume-descending (or input order for the V3 ablation)
  D3  candidate choice: eps-greedy — lexicographically best (z, y, x)
      ("bottom-left-back") with probability 1-eps, uniform random feasible
      candidate otherwise.  eps = 0 -> fully deterministic (parity and
      fixed-work regimes).
  D4' stability is NOT enforced during placement (hard = C1 + C2 + C3 only);
      a completed packing is accepted only if its stability score (fraction
      of boxes with FSR >= 1) reaches theta.  If no packing of a run reaches
      theta, the best overall packing is reported with an honest fallback
      flag (the JSON carries it).
  D5  FSR load is transitive: each box's load (own mass + everything above,
      however high) is distributed to its supports proportionally to contact
      area.
  D6  mu = 0.2 (pre-wrap interface)
  D7  tau = 0.5 (bottom-face support threshold, C3)
  D8  m = rho * V, rho = 50 kg/m3 (cancels in FSR; reported only)

Parity: with eps = 0 and one pass the decoder is fully
deterministic; the GPU implementation must produce identical placements
(`--json` output is the diff target).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass

from plp.geometry import Box
from plp.parser import Instance, allowed_orientations, load_instance

# approved parameters (spec D6/D7/D4'/D8/D3)
MU = 0.2
A_LAT_G = 0.3      # a_lat = 0.3 g; g cancels in FSR
TAU = 0.5
THETA = 0.9
RHO = 50.0         # kg/m3
EPS_QUALITY = 0.3


@dataclass(frozen=True)
class PlacedBox(Box):
    item_type: int = 0    # index into Instance.items


@dataclass(frozen=True)
class Packing:
    placed: tuple
    n_boxes: int
    volume: int
    score: float


@dataclass(frozen=True)
class RunResult:
    placements: tuple      # (item_type, x, y, z, w, d, h) per placed box
    n_boxes: int
    volume: int
    score: float
    fallback: bool         # True: no packing reached theta (D4' honest flag)
    n_passes: int
    time_s: float
    seed: int


def orientations(item) -> list[tuple[int, int, int]]:
    """Unique (w, d, h) variants in parser order (a cube yields one)."""
    seen = set()
    out = []
    for o in allowed_orientations(item):
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def _inside(p, b: Box) -> bool:
    return b.x < p[0] < b.x + b.w and b.y < p[1] < b.y + b.d and b.z < p[2] < b.z + b.h


def candidate_points(placed) -> set[tuple[int, int, int]]:
    """Extreme points: origin + 8 corners of each placed box, minus any point
    strictly inside a placed box."""
    pts = {(0, 0, 0)}
    for b in placed:
        for cx in (b.x, b.x + b.w):
            for cy in (b.y, b.y + b.d):
                for cz in (b.z, b.z + b.h):
                    pts.add((cx, cy, cz))
    if placed:
        pts = {p for p in pts if not any(_inside(p, b) for b in placed)}
    return pts


def mass_kg(volume_cm3: float) -> float:
    """D8: m = rho * V, V converted cm3 -> m3."""
    return RHO * volume_cm3 / 1e6


def compute_fsr(placed) -> list[float]:
    """Per-box FSR on a completed packing (D4'/D5/D6).

    FSR_i = mu * W_load_i / (m_i * a_lat); with W_load_i = g * load_i and
    a_lat = A_LAT_G * g:  FSR_i = (MU / A_LAT_G) * load_i / m_i.

    load_i = own mass + everything above (transitive).  A box above always
    sits at strictly higher z, so processing in descending z is a valid
    topological order; each box's load flows to its supports in proportion
    to contact area (D5).
    """
    n = len(placed)
    m = [mass_kg(b.w * b.d * b.h) for b in placed]
    load = list(m)
    for k in sorted(range(n), key=lambda i: -placed[i].z):
        bk = placed[k]
        areas = []
        for j in range(n):
            if j == k or placed[j].z + placed[j].h != bk.z:
                continue
            ox = min(bk.x + bk.w, placed[j].x + placed[j].w) - max(bk.x, placed[j].x)
            if ox <= 0:
                continue
            oy = min(bk.y + bk.d, placed[j].y + placed[j].d) - max(bk.y, placed[j].y)
            if oy > 0:
                areas.append((j, ox * oy))
        total = sum(a for _, a in areas)
        if total > 0:
            for j, a in areas:
                load[j] += load[k] * a / total
        # z > 0 with no contact cannot happen: C3 guarantees support at
        # placement and support only ever grows.
    return [(MU / A_LAT_G) * load[i] / m[i] for i in range(n)]


def stability_score(placed) -> float:
    """Fraction of placed boxes with FSR >= 1 (the paper's stability score)."""
    if not placed:
        return 0.0
    return sum(1 for f in compute_fsr(placed) if f >= 1.0) / len(placed)


def is_accepted(score: float, theta) -> bool:
    """D4': theta None (V2) accepts everything; else score >= theta (inclusive)."""
    return theta is None or score >= theta


def select_best(packings: list[Packing], theta) -> tuple[Packing, bool]:
    """D4' acceptance: highest-volume packing among those accepted; if none is
    accepted, the overall best volume with fallback=True (honest flag).
    Ties keep the earliest packing (deterministic)."""
    accepted = [p for p in packings if is_accepted(p.score, theta)]
    pool = accepted if accepted else list(packings)
    best = pool[0]
    for p in pool[1:]:
        if (p.volume, p.n_boxes) > (best.volume, best.n_boxes):
            best = p
    return best, not accepted


def item_sequence(instance: Instance, order: str = "volume") -> list[tuple[int, int]]:
    """(type, copy) pairs.  'volume': D2 volume-desc, deterministic tie-break
    (type, copy); 'input': file order (V3 ablation)."""
    seq = [(ti, c) for ti, it in enumerate(instance.items) for c in range(it.qty)]
    if order == "volume":
        seq.sort(key=lambda tc: (-instance.items[tc[0]].volume, tc[0], tc[1]))
    elif order != "input":
        raise ValueError(f"unknown order {order!r}")
    return seq


def decode_pass(instance: Instance, order: str = "volume", first_fit: bool = False,
                eps: float = 0.0, rng: random.Random | None = None) -> list[PlacedBox]:
    """One decode pass: place each item (at most once) at a feasible
    extreme-point candidate.  Hard constraints at placement: C1 + C2 + C3.
    An item with no feasible candidate is skipped; the pass continues."""
    L, W, H = instance.container_length, instance.container_width, instance.container_height
    orients = [orientations(it) for it in instance.items]
    placed: list[PlacedBox] = []
    fast: list[tuple] = []     # (x, y, z, x+w, y+d, z+h)
    cand = {(0, 0, 0)}
    for (ti, _copy) in item_sequence(instance, order):
        ors = orients[ti]
        if not ors:
            continue
        feas: list[tuple[int, int, int, int]] = []
        for p in sorted(cand, key=lambda q: (q[2], q[1], q[0])):
            x, y, z = p
            for oi, (w, d, h) in enumerate(ors):
                if x + w > L or y + d > W or z + h > H:
                    continue                            # C2
                if z > 0:
                    s = 0
                    for (bx, by, _bz, bx2, by2, bz2) in fast:
                        if bz2 == z:
                            ox = min(x + w, bx2) - max(x, bx)
                            if ox > 0:
                                oy = min(y + d, by2) - max(y, by)
                                if oy > 0:
                                    s += ox * oy
                    if s / (w * d) < TAU:
                        continue                        # C3
                ok = True
                for (bx, by, bz, bx2, by2, bz2) in fast:
                    if (x < bx2 and bx < x + w and y < by2 and by < y + d
                            and z < bz2 and bz < z + h):
                        ok = False
                        break                           # C1
                if ok:
                    feas.append((x, y, z, oi))
        if not feas:
            continue                                    # skip this copy
        if not (first_fit or eps == 0 or rng is None):
            if rng.random() >= 1.0 - eps:
                x, y, z, oi = feas[rng.randrange(len(feas))]
            else:
                x, y, z, oi = feas[0]
        else:
            x, y, z, oi = feas[0]
        w, d, h = ors[oi]
        placed.append(PlacedBox(x=x, y=y, z=z, w=w, d=d, h=h, item_type=ti))
        fast.append((x, y, z, x + w, y + d, z + h))
        # extreme-point update: old points swallowed by the new box are gone;
        # the new box's 8 corners join (they can never be strictly inside an
        # older box — that would contradict C1 at its own placement).
        cand = {p for p in cand
                if not (x < p[0] < x + w and y < p[1] < y + d and z < p[2] < z + h)}
        for cx in (x, x + w):
            for cy in (y, y + d):
                for cz in (z, z + h):
                    cand.add((cx, cy, cz))
    return placed


def run(instance: Instance, seed: int, time_limit_s: float,
        eps: float = EPS_QUALITY, theta=THETA, order: str = "volume",
        first_fit: bool = False, single_pass: bool = False) -> RunResult:
    """One experimental run: decode passes until the wall budget runs out
    (or exactly one pass in fixed-work / parity mode).  The best packing
    under the D4' acceptance rule is returned."""
    rng = random.Random(seed)
    t0 = time.monotonic()
    packings: list[Packing] = []
    k = 0
    while True:
        placed = decode_pass(instance, order=order, first_fit=first_fit, eps=eps, rng=rng)
        k += 1
        packings.append(Packing(tuple(placed), len(placed),
                                sum(b.w * b.d * b.h for b in placed),
                                stability_score(placed)))
        if single_pass or (time.monotonic() - t0) >= time_limit_s:
            break
    best, fallback = select_best(packings, theta)
    return RunResult(
        placements=tuple((b.item_type, b.x, b.y, b.z, b.w, b.d, b.h) for b in best.placed),
        n_boxes=best.n_boxes,
        volume=best.volume,
        score=best.score,
        fallback=fallback,
        n_passes=k,
        time_s=time.monotonic() - t0,
        seed=seed,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="plp.solver",
        description="3D-PLP CPU baseline decoder (V1). One instance per call.",
    )
    ap.add_argument("instance", help="path to a single-instance file (P == 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=float, default=300.0, help="wall budget per run (s)")
    ap.add_argument("--eps", type=float, default=EPS_QUALITY)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--theta", type=float, default=THETA, help="D4' acceptance threshold")
    grp.add_argument("--no-theta", dest="theta", action="store_const", const=None,
                     help="disable acceptance filter (V2)")
    ap.add_argument("--order", choices=["volume", "input"], default="volume")
    ap.add_argument("--first-fit", action="store_true", help="V3: first feasible candidate")
    ap.add_argument("--single-pass", action="store_true",
                    help="fixed-work / parity regime: exactly 1 pass (use --eps 0)")
    ap.add_argument("--json", dest="as_json", action="store_true")
    a = ap.parse_args(argv)

    inst = load_instance(a.instance)
    r = run(inst, seed=a.seed, time_limit_s=a.time, eps=a.eps, theta=a.theta,
            order=a.order, first_fit=a.first_fit, single_pass=a.single_pass)
    if a.as_json:
        print(json.dumps({
            "n_boxes": r.n_boxes,
            "volume_cm3": r.volume,
            "stability_score": r.score,
            "fallback": r.fallback,
            "n_passes": r.n_passes,
            "time_s": r.time_s,
            "seed": r.seed,
            "placements": [list(p) for p in r.placements],
        }))
    else:
        print(f"n_boxes={r.n_boxes} volume_cm3={r.volume} "
              f"stability_score={r.score:.4f} fallback={str(r.fallback).lower()} "
              f"n_passes={r.n_passes} time_s={r.time_s:.3f}")


if __name__ == "__main__":
    main()
