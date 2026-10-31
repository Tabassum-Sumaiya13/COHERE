"""CONFUSION ANALYSIS OF THE ZERO-F1 CLASSES - where a missed class's cells actually go.

FINDINGS.md section 3 counted 17 of 73 scored classes at exactly F1 = 0 and flagged this as the
cheapest high-value experiment left undone (section 9). This script reads the cached shipped
checkpoints (classifier_proto2_fold_*.pt) and their saved prediction tables - no training, no new
computation of scores - and, for every zero-F1 class, asks two questions:

  1. TRAINING COVERAGE. Does any training cohort in this fold's own label space contribute cells
     to that exact cluster? If none does, the class was mathematically unlearnable (no positive
     example existed anywhere the classifier could see one) - a roster/coverage failure, consistent
     with FINDINGS.md section 0 (class_cover r=+0.753).
  2. WHERE THE CELLS ACTUALLY GO. The row-normalised destination of true cells of that class,
     read straight from the saved prediction parquet. This distinguishes:
       - "absorbed by a covered sibling"   -> destination cluster IS covered by training cohorts
                                               and shares lineage markers with the true cluster
       - "swallowed by a dominant class"   -> destination cluster is covered and is simply the
                                               largest class in that fold, unrelated markers
       - "no consistent destination"       -> predictions scatter with no dominant target

Reads only work/ckpt/, reports/predictions/ and work/spaces/. Writes diagnostics/results/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CT = os.path.join(ROOT, "celltype_transfer")
sys.path.insert(0, CT)

import pandas as pd
import torch

from config import WORK, TRAIN
import metrics
import train_prototype_classifier as classifier

CKPT = os.path.join(WORK, "ckpt")
PRED = os.path.join(ROOT, "reports", "predictions")
SPACES = os.path.join(WORK, "spaces")

rows = []       # one row per zero-F1 class
dest_rows = []  # one row per (zero-F1 class, destination cluster) with share >= 0.03
check = []      # reproduction check against results.md's published per-fold table

for c in TRAIN:
    ckpt_path = os.path.join(CKPT, f"classifier_proto2_fold_{c}.pt")
    pred_path = os.path.join(PRED, f"proto2_fold_{c}.parquet")
    space_path = os.path.join(SPACES, f"label_map_loco_{c}.csv")
    if not (os.path.exists(ckpt_path) and os.path.exists(pred_path) and os.path.exists(space_path)):
        print(f"  {c}: missing an input, skipped")
        continue

    r = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    names = r["names"]
    # r['per_cls'] was saved from f1_all (NO unreliable-cluster drop) - it is NOT the headline's
    # class set. Recompute the CORE table the same way loco_run computes f1_core itself, so the
    # zero-F1 classes counted here are exactly the ones results.md's Section 4.6.4 table counts.
    L, _, _ = classifier.space_for(c, "fold")
    pred = pd.read_parquet(pred_path, columns=["y_true", "y_pred"])
    f1_core, per_cls = metrics.macro_f1(pred.y_true.to_numpy(), pred.y_pred.to_numpy(),
                                        L["n"], drop=L["unreliable"])
    check.append(dict(held=c, f1_core_recomputed=round(f1_core, 4),
                      f1_core_cached=round(r["f1_core"], 4),
                      match=abs(f1_core - r["f1_core"]) < 1e-6,
                      scored_classes=len(per_cls), zero_f1=int((per_cls.f1 == 0).sum())))

    zero = per_cls[per_cls.f1 == 0.0]
    if zero.empty:
        continue

    lm = pd.read_csv(space_path)

    # training coverage: cells contributed to each cluster by cohorts OTHER than the held one
    train_cov = (lm[lm.cohort != c].groupby("cluster")
                 .agg(train_n_cells=("n_cells", "sum"),
                      train_cohorts=("cohort", lambda s: sorted(set(s))))
                 .to_dict("index"))
    # size of every cluster's predicted footprint in THIS fold's own test pool, to flag "dominant"
    pred_share = pred.y_pred.value_counts(normalize=True)

    # the held cohort's own native labels feeding each cluster, for a human-readable name
    held_labels = (lm[lm.cohort == c].groupby("cluster")
                   .agg(native_labels=("label", lambda s: "; ".join(s)))
                   .to_dict("index"))

    for row in zero.itertuples():
        k = int(row.cluster)
        cov = train_cov.get(k)
        cell_rows = pred[pred.y_true == k]
        dest = cell_rows.y_pred.value_counts(normalize=True)
        top = dest.head(3)

        covered = cov is not None and cov["train_n_cells"] > 0
        top_dest_k = int(top.index[0]) if len(top) else None
        top_dest_covered = (top_dest_k is not None and top_dest_k in train_cov
                            and train_cov[top_dest_k]["train_n_cells"] > 0)
        top_dest_is_dominant = (top_dest_k is not None
                                and pred_share.get(top_dest_k, 0) >= pred_share.median() * 2
                                and pred_share.get(top_dest_k, 0) == pred_share.max())

        if not covered:
            mode = "uncovered - no training cohort has this cluster"
        elif top_dest_k is not None and top_dest_is_dominant:
            mode = "swallowed by the fold's dominant predicted class"
        elif len(top) and top.iloc[0] < 0.34:
            mode = "no consistent destination (scattered)"
        else:
            mode = "absorbed by a specific covered sibling"

        rows.append(dict(
            held=c, cluster=k, cluster_name=names[k] if k < len(names) else "?",
            native_labels=held_labels.get(k, {}).get("native_labels", ""),
            support=int(row.support),
            train_cohorts_covering=", ".join(cov["train_cohorts"]) if covered else "NONE",
            train_cells_covering=int(cov["train_n_cells"]) if cov else 0,
            top_dest_cluster=top_dest_k,
            top_dest_name=names[top_dest_k] if top_dest_k is not None and top_dest_k < len(names) else "",
            top_dest_share=round(float(top.iloc[0]), 3) if len(top) else 0.0,
            top_dest_covered=top_dest_covered,
            failure_mode=mode,
        ))
        for dk, share in top.items():
            if share < 0.03:
                continue
            dest_rows.append(dict(held=c, cluster=k, dest_cluster=int(dk),
                                  dest_name=names[int(dk)] if int(dk) < len(names) else "?",
                                  share=round(float(share), 3)))

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, "results", "confusion_zero_f1.csv"), index=False)
dd = pd.DataFrame(dest_rows)
dd.to_csv(os.path.join(HERE, "results", "confusion_zero_f1_destinations.csv"), index=False)

print(f"\n{len(d)} zero-F1 classes across {d.held.nunique()} folds\n")
print(d.failure_mode.value_counts().to_string())
print()
with pd.option_context("display.width", 200, "display.max_colwidth", 40):
    print(d[["held", "cluster", "cluster_name", "support", "train_cohorts_covering",
             "top_dest_name", "top_dest_share", "failure_mode"]].to_string(index=False))
