"""FEW-SHOT CURVE ON HARMONIZED LABELS - corrects diagnostics/fewshot.py's comparison flaw.

fewshot.py (2026-09-13) scored NATIVE labels, but the LOCO headline (0.3151) scores HARMONIZED
fold-local clusters - not the same task (FINDINGS.md section 4, flaw 1). This script samples and
scores on exactly the target LOCO uses: the label space, the class filter (unreliable clusters
dropped) and the embedding are all taken from the shipped classifier_proto2_fold_{c}.pt
checkpoints, so every row is directly comparable to that checkpoint's own f1_core.

Honest setting, unchanged from fewshot.py: the embedding is the fold-local encoder (it never saw
the held-out cohort during training); N cells per class are drawn from the cohort's TRAIN patients
only; scoring is on its TEST patients, with the same 5-seed sampling.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CT = os.path.join(ROOT, "celltype_transfer")
sys.argv = [sys.argv[0]]
os.chdir(CT)
sys.path.insert(0, CT)

import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

from config import WORK, TRAIN
import build_marker_vocabulary as vocab
import train_adversarial_encoder as adversarial
import train_prototype_classifier as classifier
import metrics

CKPT = os.path.join(WORK, "ckpt")
SEED = 20260810
SHOTS = [1, 2, 5, 10, 25, 50]
NSEED = 5

cohorts = adversarial.available()
triples, tri2idx, per, genes, ncoh = vocab.read_panel(cohorts)
V = len(triples)
excl = classifier.excluded_pairs()

rows = []
for c in TRAIN:
    ckpt_path = os.path.join(CKPT, f"classifier_proto2_fold_{c}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  {c}: no cached classifier_proto2_fold checkpoint, skipped")
        continue
    r = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    L, _, _ = classifier.space_for(c, "fold")
    m = classifier.Stage6(V, L["n"], head="proto")
    m.load_state_dict(r["state"])
    m.eval()

    d = classifier.load_cohort(c, per[c], tri2idx, V, L, excl)
    if d is None:
        print(f"  {c}: load_cohort returned None, skipped")
        continue

    def enc(part):
        U = d["U"][part]
        Z = []
        with torch.no_grad():
            for i in range(0, len(U), 1024):
                z, _, _ = m(U[i:i + 1024], d["idx"], d["present"])
                Z.append(z.cpu())
        return torch.cat(Z).numpy() if Z else np.zeros((0, 128), "float32")

    Xtr, Xte = enc("train"), enc("test")
    ytr = d["y"]["train"].cpu().numpy()
    yte = d["y"]["test"].cpu().numpy()
    classes = np.unique(ytr)
    idx_by_class = {k: np.flatnonzero(ytr == k) for k in classes}

    f1_zero_shot = r["f1_core"]  # this checkpoint's own cached LOCO score, same fold, same space

    line = [f"  {c:10} k={len(classes):>2} (0-shot LOCO {f1_zero_shot:.4f})"]
    for n in SHOTS:
        got = []
        for s in range(NSEED):
            rng = np.random.default_rng(SEED + s)
            pick = np.concatenate([rng.choice(v_, min(n, len(v_)), replace=False)
                                   for v_ in idx_by_class.values()])
            if len(np.unique(ytr[pick])) < 2:
                continue
            lr = LogisticRegression(max_iter=3000)
            lr.fit(Xtr[pick], ytr[pick])
            yp = lr.predict(Xte)
            f1, _ = metrics.macro_f1(yte, yp, L["n"], drop=L["unreliable"])
            got.append(f1)
        if got:
            rows.append(dict(cohort=c, k=len(classes), shots=n,
                             f1=round(float(np.mean(got)), 4),
                             sd=round(float(np.std(got)), 4),
                             f1_zero_shot=round(float(f1_zero_shot), 4),
                             cells=n * len(classes)))
            line.append(f"n={n}:{np.mean(got):.3f}")
    print("  ".join(line), flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, "fewshot_harmonized.csv"), index=False)
piv = d.pivot_table(index="shots", columns="cohort", values="f1")
piv["MEAN"] = piv.mean(axis=1)
print("\ncore macro-F1 by shots per HARMONIZED class (embedding from the fold that HELD THIS COHORT OUT)")
print(piv.round(4).to_string())

zero_ref = d.groupby("cohort").f1_zero_shot.first()
print(f"\nzero-shot LOCO pipeline baseline (mean of the 7 per-fold f1_core) = {zero_ref.mean():.4f}")

# cells-per-type needed to match each cohort's own zero-shot score
print("\ncells/type needed to match the zero-shot pipeline on the SAME fold (harmonized target):")
for c in zero_ref.index:
    sub = d[d.cohort == c].sort_values("shots")
    reach = sub[sub.f1 >= zero_ref[c]]
    msg = f"{reach.shots.iloc[0]}" if not reach.empty else f">{sub.shots.max()}"
    print(f"  {c:10} zero-shot={zero_ref[c]:.4f}  needs {msg} cells/type")
