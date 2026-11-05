"""Automatic intention labels for Argoverse 2 Motion Forecasting.

Labels the focal agent's maneuver over the future horizon (6 s by default)
from its ground-truth future and the AV2 lane graph.

Classes
  0 STRAIGHT      keeps its lane (or the lane's successors)
  1 LEFT_TURN     heading change > +TURN_DEG
  2 RIGHT_TURN    heading change < -TURN_DEG
  3 LANE_CHANGE_L ends on a lane reached through a left neighbour
  4 LANE_CHANGE_R ends on a lane reached through a right neighbour
  5 U_TURN        |heading change| > UTURN_DEG
  6 STOP          moves less than STOP_DISP_M over the horizon
  7 OTHER         no lane match, pedestrians, unclear cases

Rules start from Nadeem's heuristic_labeling.py (AV2 Sensor data). The main
change is that lane changes come from the lane graph (neighbours and
successors) instead of a heading-change band, which is unreliable because a
lane change barely changes the heading.

Usage
  python intention_labels.py --data_root /path/to/av2 --split val --out data/intention_labels/val.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
from collections import Counter, deque
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

CLASSES = ["STRAIGHT", "LEFT_TURN", "RIGHT_TURN", "LANE_CHANGE_L",
           "LANE_CHANGE_R", "U_TURN", "STOP", "OTHER"]

HIST_STEPS = 50          # 5 s at 10 Hz; index 49 = "now"
TURN_DEG = 30.0          # Nadeem used 20 deg over 3 s; 6 s needs a wider margin
UTURN_DEG = 135.0
STOP_DISP_M = 2.0        # under 2 m in 6 s ~ 0.33 m/s average
LANE_MATCH_M = 2.5       # max distance from a lane centreline to count as "on" it
LANE_HEADING_DEG = 60.0  # lane direction must roughly agree with the agent heading
LATERAL_M = 1.75         # fallback: sideways offset that counts as a lane change
GRAPH_DEPTH = 8          # successor hops searched for the end lane

VEHICLE_TYPES = {"vehicle", "bus", "motorcyclist", "cyclist"}


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def lane_candidates(am: ArgoverseStaticMap, xy: np.ndarray, heading: float) -> list[int]:
    """Lanes whose centreline is within LANE_MATCH_M of xy and points roughly the
    same way, closest first. At an intersection entrance several lanes overlap
    (straight and turn connectors), so all of them are kept."""
    out = []
    for seg in am.get_nearby_lane_segments(xy, 10.0):
        cl = am.get_lane_segment_centerline(seg.id)[:, :2]
        a, b = cl[:-1], cl[1:]
        ab = b - a
        # distance to each centreline piece, not just to its sample points
        t = np.clip(((xy - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
        d = np.linalg.norm(a + t[:, None] * ab - xy, axis=1)
        i = int(d.argmin())
        lane_dir = math.degrees(math.atan2(ab[i, 1], ab[i, 0]))
        if abs(wrap_deg(lane_dir - math.degrees(heading))) > LANE_HEADING_DEG:
            continue
        if d[i] < LANE_MATCH_M:
            out.append((float(d[i]), seg.id))
    return [lid for _, lid in sorted(out)]


def lane_match(am: ArgoverseStaticMap, xy: np.ndarray, heading: float):
    """Closest lane that points roughly the same way, or None."""
    c = lane_candidates(am, xy, heading)
    return c[0] if c else None


def reachable(am: ArgoverseStaticMap, start: int, depth: int = GRAPH_DEPTH) -> set[int]:
    """Lanes reachable from start by following successors only."""
    seen, q = {start}, deque([(start, 0)])
    while q:
        lid, n = q.popleft()
        if n == depth:
            continue
        for s in am.get_lane_segment_successor_ids(lid) or []:
            if s not in seen and s in am.vector_lane_segments:
                seen.add(s)
                q.append((s, n + 1))
    return seen


def side_reachable(am: ArgoverseStaticMap, start: int, side: str) -> set[int]:
    """Lanes reachable after one or two lateral moves to `side`, then successors.

    The lateral move may happen after any number of successor hops (a lane
    change that starts later in the horizon)."""
    out = set()
    for lid in reachable(am, start):
        cur = lid
        for _ in range(2):
            nb = (am.get_lane_segment_left_neighbor_id(cur) if side == "L"
                  else am.get_lane_segment_right_neighbor_id(cur))
            if nb is None or nb not in am.vector_lane_segments:
                break
            out |= reachable(am, nb)
            cur = nb
    return out


def label_agent(track, am: ArgoverseStaticMap, horizon: int = 60) -> tuple[int, dict]:
    steps = {s.timestep: s for s in track.object_states}
    now, end = HIST_STEPS - 1, HIST_STEPS - 1 + horizon
    info = {}
    if now not in steps or end not in steps:
        return CLASSES.index("OTHER"), {"reason": "missing now/end state"}

    p0, p1 = np.array(steps[now].position), np.array(steps[end].position)
    h0, h1 = steps[now].heading, steps[end].heading
    disp = float(np.linalg.norm(p1 - p0))
    dh = wrap_deg(math.degrees(h1 - h0))
    info.update(disp_m=round(disp, 2), dheading_deg=round(dh, 1))

    if disp < STOP_DISP_M:
        return CLASSES.index("STOP"), info
    if track.object_type.value not in VEHICLE_TYPES:
        return CLASSES.index("OTHER"), {**info, "reason": "not a vehicle-like agent"}
    if abs(dh) > UTURN_DEG:
        return CLASSES.index("U_TURN"), info
    if dh > TURN_DEG:
        return CLASSES.index("LEFT_TURN"), info
    if dh < -TURN_DEG:
        return CLASSES.index("RIGHT_TURN"), info

    c0, c1 = lane_candidates(am, p0, h0), lane_candidates(am, p1, h1)
    l0, l1 = (c0[0] if c0 else None), (c1[0] if c1 else None)
    info.update(lane_start=l0, lane_end=l1)
    if l0 is None or l1 is None:
        return CLASSES.index("OTHER"), {**info, "reason": "no lane match"}
    # Any overlapping start lane may be the real one (e.g. the straight connector
    # next to a turn connector), so check them all; straight wins.
    # ...but not the lanes beside it: a car already drifting over the line
    # would otherwise make a real lane change look straight.
    nb = {am.get_lane_segment_left_neighbor_id(l0), am.get_lane_segment_right_neighbor_id(l0)}
    c0 = [l0] + [c for c in c0[1:] if c not in nb]
    ends = set(c1)
    if any(ends & reachable(am, s) for s in c0):
        return CLASSES.index("STRAIGHT"), info
    if any(ends & side_reachable(am, s, "L") for s in c0):
        return CLASSES.index("LANE_CHANGE_L"), info
    if any(ends & side_reachable(am, s, "R") for s in c0):
        return CLASSES.index("LANE_CHANGE_R"), info
    # The graph has gaps (often inside intersections). Fall back on the sideways
    # offset of the end point relative to the starting heading.
    fwd = np.array([math.cos(h0), math.sin(h0)])
    lat = float(np.cross(fwd, p1 - p0))  # > 0 = left of the start heading
    info.update(reason="graph gap, lateral fallback", lateral_m=round(lat, 2))
    if lat > LATERAL_M:
        return CLASSES.index("LANE_CHANGE_L"), info
    if lat < -LATERAL_M:
        return CLASSES.index("LANE_CHANGE_R"), info
    return CLASSES.index("STRAIGHT"), info


def label_dir(d: Path, horizon: int = 60) -> list:
    """One CSV row for the focal agent of the scenario folder d."""
    sc = scenario_serialization.load_argoverse_scenario_parquet(d / f"scenario_{d.name}.parquet")
    am = ArgoverseStaticMap.from_json(d / f"log_map_archive_{d.name}.json")
    focal = next(t for t in sc.tracks if t.track_id == sc.focal_track_id)
    lab, info = label_agent(focal, am, horizon)
    return [sc.scenario_id, focal.track_id, focal.object_type.value, lab, CLASSES[lab],
            info.get("disp_m"), info.get("dheading_deg"), info.get("reason", "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="intention_labels.csv")
    ap.add_argument("--horizon", type=int, default=60, help="future steps (10 Hz)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    args = ap.parse_args()

    dirs = sorted(p for p in (Path(args.data_root) / args.split).iterdir() if p.is_dir())
    work = partial(label_dir, horizon=args.horizon)
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            rows = list(tqdm(pool.imap(work, dirs, chunksize=16), total=len(dirs)))
    else:
        rows = [work(d) for d in tqdm(dirs)]
    counts = Counter(r[4] for r in rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "track_id", "object_type", "label", "label_name",
                    "disp_m", "dheading_deg", "note"])
        w.writerows(rows)
    total = sum(counts.values())
    for c in CLASSES:
        print(f"{c:14s} {counts[c]:6d}  {100 * counts[c] / max(total, 1):5.1f}%")


if __name__ == "__main__":
    main()
