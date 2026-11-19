"""Intention from DeMo's own paths: a small model on top of the rule baseline.

The rule baseline (analyze_intentions.py, section 5) applies the label rules to
DeMo's most likely path and already beats the intention heads. This script
learns on top of it, with DeMo frozen:

  extract  run DeMo once on a split; label all 6 predicted paths with the rules;
           save per-scenario features (rule result per path, path
           probabilities, the heads' output, agent type, path geometry)
  fit      train a small classifier (gradient-boosted trees) on the train
           features and compare it on val with the rule, the probability-
           weighted rule over all 6 paths, and the heads
  diagnose per true maneuver, split DeMo's failures into
           A no path has the right maneuver      (mode generation)
           B a right path exists but ranks low   (ranking / calibration)
           C the right path is top but is far off (geometry / refinement)
           and show which maneuver each of the 6 mode queries stands for

Usage
  python fusion_intent.py extract --ckpt CKPT --data_root PROCESSED --raw_root RAW \
      --intention_dir LABELS --split train --out fusion_train.npz
  python fusion_intent.py extract ... --split val --out fusion_val.npz
  python fusion_intent.py fit --train fusion_train.npz --val fusion_val.npz
  python fusion_intent.py diagnose --val fusion_val.npz
"""
import argparse
import csv
import multiprocessing
import os
from pathlib import Path

import numpy as np

from intention_labels import CLASSES, VEHICLE_TYPES, label_states

C = len(CLASSES)
TYPES = ["vehicle", "bus", "motorcyclist", "cyclist", "pedestrian"]


def _label_paths(job):
    """Rule label, heading change, displacement and off-path distance for each of the 6 paths."""
    from av2.map.map_api import ArgoverseStaticMap
    sid, raw_root, split, p0, h0, paths, otype = job
    am = ArgoverseStaticMap.from_json(Path(raw_root) / split / sid / f"log_map_archive_{sid}.json")
    out = []
    for path in paths:
        d = path[-1] - path[-6]                                        # last 0.5 s
        h1 = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 0.2 else h0
        lab, info = label_states(am, p0, h0, path[-1], h1, otype)
        out.append((lab, info.get("dheading_deg", 0.0), info.get("disp_m", 0.0),
                    info.get("off_path_m") or 0.0))
    return out


def extract(args):
    import torch
    from torch.utils.data import DataLoader
    from src.datamodule.av2_dataset import Av2Dataset, collate_fn
    from src.model.model_forecast import ModelForecast

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ModelForecast(embed_dim=128, future_steps=60, num_heads=8, mlp_ratio=4.0,
                        qkv_bias=False, drop_path=0.2, num_intentions=C)
    res = net.load_from_checkpoint(args.ckpt)
    if res.missing_keys:
        raise SystemExit(f"checkpoint is missing {len(res.missing_keys)} weights")
    net.to(dev).eval()

    ds = Av2Dataset(data_root=args.data_root, split=args.split, intention_dir=args.intention_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)
    types = {}
    with open(Path(args.intention_dir) / f"{args.split}.csv") as f:
        for r in csv.DictReader(f):
            types[r["scenario_id"]] = r["object_type"]

    sids, labels, pis, heads, jobs, fdes, ades = [], [], [], [], [], [], []
    from tqdm.auto import tqdm
    with torch.no_grad():
        for batch in tqdm(dl, desc="DeMo"):
            data = batch[-1] if isinstance(batch, list) else batch
            data = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in data.items()}
            out = net(data)
            y = out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]   # [B, K, 60, 2]
            pi = (out["new_pi"] if out.get("new_pi") is not None else out["pi"]).softmax(-1)
            th = data["theta"].double()
            rot = torch.stack([torch.stack([th.cos(), -th.sin()], -1),
                               torch.stack([th.sin(), th.cos()], -1)], -2)            # [B, 2, 2]
            world = (y.double() @ rot[:, None].transpose(-1, -2)
                     + data["origin"].double()[:, None, None]).cpu().numpy()          # [B, K, 60, 2]
            hp = out["intent_logits"].softmax(-1).cpu().numpy()
            gt = data["target"][:, 0]                                                   # [B, 60, 2], agent frame
            fde = (y[:, :, -1] - gt[:, None, -1]).norm(dim=-1).cpu().numpy()           # [B, K]
            ade = (y - gt[:, None]).norm(dim=-1).mean(-1).cpu().numpy()                # [B, K]
            for i, sid in enumerate(data["scenario_id"]):
                lab = int(data["intention"][i])
                if lab < 0:
                    continue
                sids.append(sid); labels.append(lab)
                pis.append(pi[i].cpu().numpy()); heads.append(hp[i])
                fdes.append(fde[i]); ades.append(ade[i])
                jobs.append((sid, args.raw_root, args.split, data["origin"][i].double().cpu().numpy(),
                             float(data["theta"][i]), world[i], types.get(sid, "vehicle")))

    with multiprocessing.Pool(args.workers) as pool:
        per_path = list(tqdm(pool.imap(_label_paths, jobs, chunksize=16), total=len(jobs), desc="rules"))
    np.savez_compressed(
        args.out, scenario_id=np.array(sids), label=np.array(labels),
        pi=np.array(pis, dtype=np.float32), heads=np.array(heads, dtype=np.float32),
        rule=np.array([[p[0] for p in pp] for pp in per_path], dtype=np.int64),
        geom=np.array([[p[1:] for p in pp] for pp in per_path], dtype=np.float32),
        fde=np.array(fdes, dtype=np.float32), ade=np.array(ades, dtype=np.float32),
        otype=np.array([types.get(s, "vehicle") for s in sids]))
    print(f"saved {len(sids)} scenarios to {args.out}")


def features(z):
    pi, rule, geom = z["pi"], z["rule"], z["geom"]
    order = np.argsort(-pi, axis=1)
    pi_s = np.take_along_axis(pi, order, 1)                                   # probabilities, sorted
    rule_s = np.take_along_axis(rule, order, 1)
    geom_s = np.take_along_axis(geom, order[..., None], 1)                    # [N, 6, 3]
    onehot = np.eye(C)[rule_s]                                               # [N, 6, C]
    soft = (onehot * pi_s[..., None]).sum(1)                                 # probability-weighted rule
    typ = np.stack([(z["otype"] == t) for t in TYPES], 1).astype(np.float32)
    return np.concatenate([soft, onehot[:, 0], pi_s, z["heads"], typ,
                           geom_s[:, :3].reshape(len(pi), -1)], 1), soft


def report(lab, preds):
    present = [c for c in range(C) if (lab == c).sum()]
    print(f"{'class':14s} {'n':>5s}" + "".join(f"{k:>10s}" for k in preds))
    f1s = {k: [] for k in preds}
    for c in range(C):
        n = int((lab == c).sum())
        row = f"{CLASSES[c]:14s} {n:5d}"
        for k, p in preds.items():
            tp = int(((p == c) & (lab == c)).sum())
            pr = tp / max(int((p == c).sum()), 1); rc = tp / max(n, 1)
            f1 = 2 * pr * rc / max(pr + rc, 1e-9)
            if c in present:
                f1s[k].append(f1)
            row += f"{f1:10.2f}"
        print(row)
    print(f"{'macro F1':20s}" + "".join(f"{np.mean(v):10.3f}" for v in f1s.values()))
    print(f"{'accuracy':20s}" + "".join(f"{(p == lab).mean():10.3f}" for p in preds.values()))


def fit(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr, va = np.load(args.train), np.load(args.val)
    Xtr, _ = features(tr)
    Xva, soft = features(va)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         early_stopping=True, random_state=0)
    clf.fit(Xtr, tr["label"])
    preds = {
        "rule": va["rule"][np.arange(len(va["label"])), va["pi"].argmax(1)],
        "soft rule": soft.argmax(1),
        "heads": va["heads"].argmax(1),
        "fusion": clf.predict(Xva),
    }
    print(f"train {len(tr['label'])} scenarios, val {len(va['label'])} scenarios\n")
    print("F1 per class on val (rule = labels of DeMo's top path; soft rule = probability-weighted"
          " over all 6 paths; fusion = trained on top of both)")
    report(va["label"], preds)
    np.savez_compressed(args.val.replace(".npz", "_pred.npz"), scenario_id=va["scenario_id"],
                        **{k.replace(" ", "_"): v for k, v in preds.items()})


def diagnose(args):
    """Where does DeMo fail on each maneuver? Paths stay in DeMo's mode order (mode k = query k)."""
    z = np.load(args.val)
    if "fde" not in z:
        raise SystemExit("this npz has no path errors: run extract again with the current fusion_intent.py")
    lab, rule, pi, fde = z["label"], z["rule"], z["pi"], z["fde"]
    n_all = len(lab)
    top = pi.argmax(1)
    ar = np.arange(n_all)
    right = rule == lab[:, None]                                   # [N, 6] path k has the true maneuver
    cover = right.any(1)
    top_ok = right[ar, top]
    top_fde = fde[ar, top]
    cond = np.where(right, fde, np.inf).min(1)                     # best right-maneuver path
    miss = args.miss

    print(f"{n_all} val scenarios, miss = final error > {miss} m\n")
    print("1. Per true maneuver (A + B + C + OK = 100%)")
    hdr = (f"{'maneuver':14s} {'n':>5s} {'Cov@6':>6s} {'Top1':>6s} {'minFDE6':>8s} {'condFDE':>8s} {'topFDE':>7s}"
           f" {'A gen':>6s} {'B rank':>6s} {'C geom':>6s} {'OK':>6s}")
    print(hdr)
    rows = []
    for c in range(C):
        m = lab == c
        n = int(m.sum())
        if n == 0:
            continue
        a = ~cover[m]                                              # no right path
        b = cover[m] & ~top_ok[m]                                  # right path exists, not top
        cc = top_ok[m] & (top_fde[m] > miss)                       # top is right, but far off
        ok = top_ok[m] & (top_fde[m] <= miss)
        cf = cond[m][cover[m]]
        r = dict(maneuver=CLASSES[c], n=n, cov6=cover[m].mean(), top1=top_ok[m].mean(),
                 minFDE6=fde[m].min(1).mean(), condFDE=cf.mean() if len(cf) else np.nan,
                 topFDE=top_fde[m][top_ok[m]].mean() if top_ok[m].any() else np.nan,
                 A=a.mean(), B=b.mean(), C=cc.mean(), OK=ok.mean())
        rows.append(r)
        print(f"{r['maneuver']:14s} {n:5d} {r['cov6']:6.1%} {r['top1']:6.1%} {r['minFDE6']:8.2f} "
              f"{r['condFDE']:8.2f} {r['topFDE']:7.2f} {r['A']:6.1%} {r['B']:6.1%} {r['C']:6.1%} {r['OK']:6.1%}")
    print("  Cov@6   = any of the 6 paths has the true maneuver")
    print("  Top1    = the highest-scored path has the true maneuver")
    print("  condFDE = final error of the best right-maneuver path (scenarios where one exists)")
    print("  topFDE  = final error of the top path when it has the right maneuver")
    print("  A = no right path (mode generation), B = right path exists but not top (ranking),")
    print(f"  C = top path right but final error > {miss} m (geometry / refinement)")

    print("\n2. What each mode query stands for (share of scenarios, rows = query)")
    short = ["STR", "LEFT", "RIGHT", "LC_L", "LC_R", "UTRN", "STOP", "OTH"]
    print(f"{'query':6s}" + "".join(f"{s:>7s}" for s in short) + f"{'purity':>8s}{'is top':>8s}")
    veh = np.isin(z["otype"], list(VEHICLE_TYPES))
    for k in range(rule.shape[1]):
        d = np.bincount(rule[veh, k], minlength=C) / max(int(veh.sum()), 1)
        print(f"#{k:<5d}" + "".join(f"{v:7.1%}" for v in d) + f"{d.max():8.1%}{(top == k).mean():8.1%}")
    print("  purity = share of the most common maneuver for that query (vehicle-like agents only);")
    print("  a query that always means the same maneuver has high purity")
    print("\n   Same, only for scenarios whose true maneuver is a turn or lane change:")
    sel = veh & np.isin(lab, [1, 2, 3, 4])
    for k in range(rule.shape[1]):
        d = np.bincount(rule[sel, k], minlength=C) / max(int(sel.sum()), 1)
        print(f"#{k:<5d}" + "".join(f"{v:7.1%}" for v in d) + f"{d.max():8.1%}{(top[sel] == k).mean():8.1%}")

    out = args.val.replace(".npz", "_diagnosis.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\nsaved", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--data_root", required=True, help="DeMo_processed folder holding <split>/")
    e.add_argument("--raw_root", required=True, help="raw AV2 folder holding <split>/<id>/")
    e.add_argument("--intention_dir", required=True, help="folder with <split>.csv labels")
    e.add_argument("--split", default="val", choices=["train", "val"])
    e.add_argument("--out", required=True)
    e.add_argument("--batch_size", type=int, default=32)
    e.add_argument("--workers", type=int, default=os.cpu_count())
    f = sub.add_parser("fit")
    f.add_argument("--train", required=True)
    f.add_argument("--val", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--val", required=True)
    d.add_argument("--miss", type=float, default=2.0, help="final error (m) above which a path is a miss")
    args = ap.parse_args()
    {"extract": extract, "fit": fit, "diagnose": diagnose}[args.cmd](args)


if __name__ == "__main__":
    main()
