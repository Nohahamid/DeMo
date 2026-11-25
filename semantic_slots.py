"""Which maneuvers are possible for the focal agent, from the lane graph only.

For every scenario this looks at the map around the focal agent *now* (no
future information) and writes, for each semantic maneuver,

  feasible   1 if the lane graph allows it within the 6 s horizon, else 0
  anchor     a rough end point (map frame) for that maneuver: the point on the
             target lane at about the distance the agent covers in 6 s

The model gives DeMo's six mode queries a meaning from these: one query per
feasible maneuver, the rest stay free (scene-conditioned semantic allocation).

Maneuvers (SEMANTICS) and their label ids in intention_labels.CLASSES
  STRAIGHT 0, LEFT_TURN 1, RIGHT_TURN 2, LANE_CHANGE_L 3, LANE_CHANGE_R 4, STOP 6

Usage
  python semantic_slots.py --data_root /path/to/av2 --split val --out data/semantic_slots/val.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

from intention_labels import (CLASSES, HIST_STEPS, TURN_DEG, VEHICLE_TYPES,
                              lane_candidates, wrap_deg)

SEMANTICS = ["STRAIGHT", "LEFT_TURN", "RIGHT_TURN", "LANE_CHANGE_L", "LANE_CHANGE_R", "STOP"]
SEM_LABEL = [CLASSES.index(s) for s in SEMANTICS]   # semantic slot -> label id
HORIZON_S = 6.0
MIN_DIST_M = 5.0
MAX_DIST_M = 120.0
TURN_SLACK_M = 15.0        # a turn must finish within the travel distance + this
MAX_PATHS = 64             # successor paths explored per start lane


def _cl(am, lid):
    return am.get_lane_segment_centerline(lid)[:, :2]


def _heading(cl, end=True):
    d = cl[-1] - cl[-2] if end else cl[1] - cl[0]
    return math.atan2(d[1], d[0])


def _project(cl, xy):
    """Arc length along the centreline of the point closest to xy."""
    a, b = cl[:-1], cl[1:]
    ab = b - a
    seg = np.linalg.norm(ab, axis=1)
    t = np.clip(((xy - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    d = np.linalg.norm(a + t[:, None] * ab - xy, axis=1)
    i = int(d.argmin())
    return float(seg[:i].sum() + t[i] * seg[i])


def _point_at(polyline, s):
    """Point at arc length s along a polyline (clamped to its ends)."""
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(i, len(seg) - 1)
    t = (s - cum[i]) / max(seg[i], 1e-9)
    return polyline[i] + t * (polyline[i + 1] - polyline[i])


def _paths(am, start, s0, length):
    """Successor chains from `start` that cover `length` metres beyond arc s0.
    Returns (polyline from s0 on, list of lane ids)."""
    out, stack = [], [(start, [start])]
    while stack and len(out) < MAX_PATHS:
        lid, chain = stack.pop()
        pts = np.concatenate([_cl(am, l) if k == 0 else _cl(am, l)[1:] for k, l in enumerate(chain)])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if seg.sum() - s0 >= length:
            out.append((pts, chain))
            continue
        succ = [s for s in (am.get_lane_segment_successor_ids(lid) or []) if s in am.vector_lane_segments]
        if not succ or len(chain) > 25:
            out.append((pts, chain))
            continue
        for s in succ:
            stack.append((s, chain + [s]))
    return out


def _straightest(am, start, s0, length):
    """The successor chain with the smallest total heading change."""
    best = None
    for pts, chain in _paths(am, start, s0, length):
        dh = abs(wrap_deg(math.degrees(_heading(_cl(am, chain[-1])) - _heading(_cl(am, start), end=False))))
        if best is None or dh < best[0]:
            best = (dh, pts)
    return best[1]


def _neighbour(am, lid, side, h0):
    nb = am.get_lane_segment_left_neighbor_id(lid) if side == "L" else am.get_lane_segment_right_neighbor_id(lid)
    if nb is None or nb not in am.vector_lane_segments:
        return None
    cl = _cl(am, nb)
    # same driving direction only (AV2 also lists opposite lanes as neighbours)
    if abs(wrap_deg(math.degrees(_heading(cl, end=False) - h0))) > 60.0:
        return None
    return nb


def slots_for(am, p0, h0, speed, object_type):
    """feasible [6] (0/1) and anchors [6, 2] (map frame) for one agent."""
    feas = np.zeros(len(SEMANTICS), dtype=np.int64)
    anchor = np.tile(np.asarray(p0, dtype=float), (len(SEMANTICS), 1))
    if object_type not in VEHICLE_TYPES:
        return feas, anchor, "not a vehicle-like agent"
    p0 = np.asarray(p0, dtype=float)
    feas[SEMANTICS.index("STOP")] = 1
    cands = lane_candidates(am, p0, h0)
    if not cands:
        return feas, anchor, "no lane match"
    dist = float(np.clip(speed * HORIZON_S, MIN_DIST_M, MAX_DIST_M))

    # straight: follow the lane graph along the straightest chain
    l0 = cands[0]
    s0 = _project(_cl(am, l0), p0)
    pts = _straightest(am, l0, s0, dist)
    feas[0] = 1
    anchor[0] = _point_at(pts, s0 + dist)

    # turns: any successor chain from any start candidate that turns > TURN_DEG
    # within the travel distance (+ slack)
    for c in cands:
        sc = _project(_cl(am, c), p0)
        for pts, chain in _paths(am, c, sc, dist + TURN_SLACK_M):
            cum = 0.0
            for lid in chain:
                cl = _cl(am, lid)
                cum += float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
                dh = wrap_deg(math.degrees(_heading(cl) - h0))
                if abs(dh) > TURN_DEG and abs(dh) < 135.0:
                    j = SEMANTICS.index("LEFT_TURN" if dh > 0 else "RIGHT_TURN")
                    if not feas[j]:
                        feas[j] = 1
                        anchor[j] = _point_at(pts, sc + dist)
                    break
                if cum - sc > dist + TURN_SLACK_M:
                    break

    # lane changes: a same-direction neighbour of the start lane, or of a lane
    # on the straight chain ahead (a lane change that starts later)
    ahead = [l0] + [l for l in _straightest_chain(am, l0, s0, dist)]
    for side, name in (("L", "LANE_CHANGE_L"), ("R", "LANE_CHANGE_R")):
        j = SEMANTICS.index(name)
        for lid in ahead:
            nb = _neighbour(am, lid, side, h0)
            if nb is None:
                continue
            sn = _project(_cl(am, nb), p0)
            npts = _straightest(am, nb, sn, dist)
            feas[j] = 1
            anchor[j] = _point_at(npts, sn + dist)
            break
    return feas, anchor, ""


def _straightest_chain(am, start, s0, length):
    best = None
    for pts, chain in _paths(am, start, s0, length):
        dh = abs(wrap_deg(math.degrees(_heading(_cl(am, chain[-1])) - _heading(_cl(am, start), end=False))))
        if best is None or dh < best[0]:
            best = (dh, chain)
    return best[1][1:]


def slots_dir(d: Path) -> list:
    sc = scenario_serialization.load_argoverse_scenario_parquet(d / f"scenario_{d.name}.parquet")
    am = ArgoverseStaticMap.from_json(d / f"log_map_archive_{d.name}.json")
    focal = next(t for t in sc.tracks if t.track_id == sc.focal_track_id)
    st = {s.timestep: s for s in focal.object_states}.get(HIST_STEPS - 1)
    if st is None:
        feas, anchor, note = np.zeros(len(SEMANTICS), int), np.zeros((len(SEMANTICS), 2)), "missing now state"
    else:
        try:
            feas, anchor, note = slots_for(am, st.position, st.heading, float(np.linalg.norm(st.velocity)),
                                           focal.object_type.value)
        except Exception as e:                                  # odd map: fall back to plain DeMo
            feas, anchor, note = np.zeros(len(SEMANTICS), int), np.tile(st.position, (len(SEMANTICS), 1)), f"error {e}"
    return [sc.scenario_id] + feas.tolist() + [round(float(v), 3) for v in anchor.reshape(-1)] + [note]


def header():
    return (["scenario_id"] + [f"feas_{s}" for s in SEMANTICS]
            + [f"{a}_{s}" for s in SEMANTICS for a in ("ax", "ay")] + ["note"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    dirs = sorted(p for p in (Path(args.data_root) / args.split).iterdir() if p.is_dir())
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            rows = list(tqdm(pool.imap(slots_dir, dirs, chunksize=16), total=len(dirs)))
    else:
        rows = [slots_dir(d) for d in tqdm(dirs)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header())
        w.writerows(rows)
    feas = np.array([r[1:1 + len(SEMANTICS)] for r in rows])
    print(f"{len(rows)} scenarios; share where each maneuver is feasible:")
    for s, v in zip(SEMANTICS, feas.mean(0)):
        print(f"  {s:14s} {v:6.1%}")
    print("notes:", Counter(r[-1].split(" ")[0] for r in rows if r[-1]).most_common(4))


if __name__ == "__main__":
    main()
