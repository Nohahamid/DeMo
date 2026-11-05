"""Per-maneuver and per-agent-type analysis of a DeMo + named-intention checkpoint.

Usage
  python analyze_intentions.py                      # newest outputs/**/checkpoints/last.ckpt
  python analyze_intentions.py --ckpt path/to.ckpt

Prints, on the validation set:
  1. precision / recall / F1 per intention class, with path errors per class
  2. the confusion matrix (rows = true label, columns = predicted)
  3. results per agent type (vehicle, pedestrian, ...)
  4. whether the best-matching mode is given the right name
and writes one row per scenario to <checkpoint folder>/../analysis_val.csv.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from intention_labels import CLASSES
from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from src.model.model_forecast import ModelForecast

MISS_M = 2.0


def newest_ckpt():
    c = glob.glob("outputs/**/checkpoints/last.ckpt", recursive=True)
    if not c:
        raise SystemExit("No outputs/**/checkpoints/last.ckpt found; pass --ckpt")
    return max(c, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_root", default="data/DeMo_processed")
    ap.add_argument("--intention_dir", default="data/intention_labels")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    ckpt = args.ckpt or newest_ckpt()
    print("checkpoint:", ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = ModelForecast(embed_dim=128, future_steps=60, num_heads=8, mlp_ratio=4.0,
                        qkv_bias=False, drop_path=0.2, num_intentions=len(CLASSES))
    res = net.load_from_checkpoint(ckpt)
    missing = [k for k in res.missing_keys]
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
    net.to(dev).eval()

    ds = Av2Dataset(data_root=args.data_root, split="val", intention_dir=args.intention_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)
    types = {}
    with open(os.path.join(args.intention_dir, "val.csv")) as f:
        for r in csv.DictReader(f):
            types[r["scenario_id"]] = r["object_type"]

    rows = []
    with torch.no_grad():
        for batch in dl:
            data = batch[-1] if isinstance(batch, list) else batch
            data = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in data.items()}
            out = net(data)
            y = out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]
            gt = data["target"][:, 0]                                   # [B, 60, 2]
            fde = (y[:, :, -1] - gt[:, None, -1]).norm(dim=-1)          # [B, K]
            ade = (y - gt[:, None]).norm(dim=-1).mean(-1)               # [B, K]
            best = fde.argmin(-1)
            prob = out["intent_logits"].softmax(-1)
            mode_pred = out["mode_intent_logits"].argmax(-1)            # [B, K]
            for i, sid in enumerate(data["scenario_id"]):
                rows.append(dict(
                    scenario_id=sid, object_type=types.get(sid, "?"),
                    label=int(data["intention"][i]), pred=int(prob[i].argmax()),
                    conf=float(prob[i].max()),
                    best_mode_name=int(mode_pred[i, best[i]]),
                    minFDE6=float(fde[i].min()), minADE6=float(ade[i].min()),
                ))

    lab = np.array([r["label"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    mfde = np.array([r["minFDE6"] for r in rows])
    ok = lab >= 0
    print(f"\n{ok.sum()} labelled val scenarios, overall minFDE6 {mfde.mean():.3f} m, "
          f"MR {(mfde > MISS_M).mean():.3f}")

    print("\n1. Per intention class")
    print(f"{'class':14s} {'n':>5s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'minFDE6':>8s} {'MR':>6s}")
    f1s = []
    for c, name in enumerate(CLASSES):
        n = int((lab == c).sum())
        tp = int(((pred == c) & (lab == c)).sum())
        p = tp / max(int(((pred == c) & ok).sum()), 1)
        r = tp / max(n, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if n:
            f1s.append(f1)
        sel = lab == c
        fd = f"{mfde[sel].mean():8.3f}" if n else f"{'-':>8s}"
        mr = f"{(mfde[sel] > MISS_M).mean():6.3f}" if n else f"{'-':>6s}"
        print(f"{name:14s} {n:5d} {p:6.2f} {r:6.2f} {f1:6.2f} {fd} {mr}")
    print(f"macro F1 over classes present in val: {np.mean(f1s):.3f}")

    print("\n2. Confusion matrix (rows = true, columns = predicted)")
    short = ["STR", "LEFT", "RIGHT", "LC_L", "LC_R", "UTRN", "STOP", "OTH"]
    print(" " * 6 + "".join(f"{s:>6s}" for s in short))
    for c in range(len(CLASSES)):
        print(f"{short[c]:6s}" + "".join(f"{int(((lab == c) & (pred == d)).sum()):6d}"
                                       for d in range(len(CLASSES))))

    print("\n3. Per agent type")
    by = defaultdict(list)
    for r in rows:
        by[r["object_type"]].append(r)
    print(f"{'type':12s} {'n':>5s} {'intent acc':>10s} {'minFDE6':>8s} {'MR':>6s}")
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        l = np.array([r["label"] for r in rs]); p = np.array([r["pred"] for r in rs])
        f = np.array([r["minFDE6"] for r in rs]); m = l >= 0
        acc = (l[m] == p[m]).mean() if m.any() else float("nan")
        print(f"{t:12s} {len(rs):5d} {acc:10.3f} {f.mean():8.3f} {(f > MISS_M).mean():6.3f}")

    bm = np.array([r["best_mode_name"] for r in rows])
    print(f"\n4. Best-matching mode carries the right name: {(bm[ok] == lab[ok]).mean():.3f}")

    out_csv = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "analysis_val.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\nper-scenario results:", out_csv)


if __name__ == "__main__":
    main()
