"""Draw scenarios of one intention class for a hand check of the labels.

Each panel: nearby lane centrelines (grey), the focal agent's 5 s history
(blue), its true 6 s future (green, dot = end), and the lanes the labeller
matched at the start (orange) and end (red). The title gives the label and,
if an analysis CSV exists, the model's prediction.

Usage
  python plot_intentions.py --data_root /content/av2_raw --label LANE_CHANGE_L --n 30
"""
import argparse
import csv
import glob
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

from intention_labels import CLASSES, HIST_STEPS, label_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="raw AV2 folder holding val/")
    ap.add_argument("--split", default="val")
    ap.add_argument("--label", default="LANE_CHANGE_L", choices=CLASSES)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--labels_csv", default="data/intention_labels/val.csv")
    ap.add_argument("--analysis_csv", default=None, help="default: newest outputs/**/analysis_val.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    target = CLASSES.index(args.label)

    with open(args.labels_csv) as f:
        ids = [r["scenario_id"] for r in csv.DictReader(f) if int(r["label"]) == target][: args.n]
    pred = {}
    acsv = args.analysis_csv or max(glob.glob("outputs/**/analysis_val.csv", recursive=True),
                                    key=os.path.getmtime, default=None)
    if acsv:
        with open(acsv) as f:
            pred = {r["scenario_id"]: (int(r["pred"]), float(r["conf"])) for r in csv.DictReader(f)}
    if not ids:
        raise SystemExit(f"no {args.label} scenarios in {args.labels_csv}")

    cols = 5
    rows = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, sid in zip(axes, ids):
        d = Path(args.data_root) / args.split / sid
        sc = scenario_serialization.load_argoverse_scenario_parquet(d / f"scenario_{sid}.parquet")
        am = ArgoverseStaticMap.from_json(d / f"log_map_archive_{sid}.json")
        tr = next(t for t in sc.tracks if t.track_id == sc.focal_track_id)
        st = {s.timestep: s for s in tr.object_states}
        xy = np.array([st[t].position for t in sorted(st)])
        ts = np.array(sorted(st))
        hist, fut = xy[ts < HIST_STEPS], xy[ts >= HIST_STEPS - 1]
        c = xy[ts == HIST_STEPS - 1][0] if (ts == HIST_STEPS - 1).any() else xy[len(xy) // 2]
        lab, info = label_agent(tr, am)
        for seg in am.get_nearby_lane_segments(c, 60.0):
            cl = am.get_lane_segment_centerline(seg.id)[:, :2]
            colr, lw = "#C9CDD2", 1
            if seg.id == info.get("lane_start"):
                colr, lw = "#E08A2C", 3
            if seg.id == info.get("lane_end"):
                colr, lw = "#C0392B", 3
            ax.plot(cl[:, 0], cl[:, 1], color=colr, lw=lw, zorder=1)
        ax.plot(hist[:, 0], hist[:, 1], color="#2F5DA8", lw=2.5, zorder=3)
        ax.plot(fut[:, 0], fut[:, 1], color="#2E9E5B", lw=2.5, zorder=3)
        ax.scatter(fut[-1:, 0], fut[-1:, 1], color="#2E9E5B", s=25, zorder=4)
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 30) / 2 + 10
        m = xy.mean(0)
        ax.set_xlim(m[0] - span, m[0] + span); ax.set_ylim(m[1] - span, m[1] + span)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        t = f"{sid[:8]}  label {CLASSES[lab]}"
        if sid in pred:
            p, conf = pred[sid]
            t += f"\npred {CLASSES[p]} ({conf:.2f})"
        if info.get("off_path_m") is not None:
            t += f"\nend {info['off_path_m']} m beside its own path"
        if info.get("reason"):
            t += f"\n{info['reason']}"
        ax.set_title(t, fontsize=8)
    for ax in axes[len(ids):]:
        ax.axis("off")
    fig.suptitle(f"{args.label}: {len(ids)} scenarios. Blue = past, green = true future, "
                 "orange/red = matched start/end lane", fontsize=11)
    fig.tight_layout()
    out = args.out or f"check_{args.label}.png"
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
