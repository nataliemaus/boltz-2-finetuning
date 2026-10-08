#!/usr/bin/env python3
"""Step 3: score a checkpoint against stock Boltz-2 on held-out data.

    python scripts/evaluate.py --config configs/starter.json --checkpoint runs/starter/epoch_007.pt

Reports concordance and Spearman for both, on the same records, so the comparison is paired.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boltz_ft import data as D
from boltz_ft import losses as L
from boltz_ft.model import adapter_width, load_affinity_heads, make_adapter, predict_value


def score(m1, m2, adapter, groups, have, dev):
    """Per-group concordance, plus a pooled Spearman over DISTINCT records.

    A record can belong to several comparison groups, so pooling group by group would count it
    more than once and compute the correlation over a duplicated sample.
    """
    per_group, pooled = [], {}
    cache = {}
    with torch.no_grad():
        for g in groups:
            rids = [r for r in g["points"] if r in have]
            if len(rids) < 2:
                continue
            preds = []
            for r in rids:
                if r not in cache:
                    blob = torch.load(have[r], map_location="cpu", weights_only=False)
                    cache[r] = float(predict_value(m1, m2, blob, dev, adapter))
                preds.append(cache[r])
            meas = [g["points"][r] for r in rids]
            c = L.concordance(preds, meas)
            if c == c:
                per_group.append(c)
            for r, pr, me in zip(rids, preds, meas):
                pooled.setdefault(r, (pr, me))
    flat_p = [v[0] for v in pooled.values()]
    flat_m = [v[1] for v in pooled.values()]
    rho = float(stats.spearmanr(flat_p, flat_m).statistic) if len(flat_p) > 2 else float("nan")
    return {"groups": len(per_group), "records": len(pooled),
            "concordance": float(np.mean(per_group)) if per_group else float("nan"),
            "spearman": rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="write the numbers here as JSON")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    dev = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    rows = D.load_measurements(cfg["measurements"])
    have = D.find_blobs(cfg["blob_dirs"])
    _, va_rows = D.split_records(rows, cfg.get("val_frac", 0.1), cfg.get("val_by"),
                                 cfg.get("seed", 0))
    groups = D.build_groups(va_rows, have, cfg["ranking_groups"], cfg.get("min_delta", 0.3))
    if not groups:
        sys.exit("no validation group has blobs; nothing to evaluate")

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    base1, base2 = load_affinity_heads(cfg["affinity_ckpt"], dev)
    tuned1, tuned2 = load_affinity_heads(cfg["affinity_ckpt"], dev)
    tuned1.load_state_dict(ck["m1"]); tuned2.load_state_dict(ck["m2"])
    acfg = ck.get("adapter_cfg", {})
    adapter = make_adapter(dev, hidden=acfg.get("hidden", 512), d_s=acfg.get("d_s", 384))
    adapter.load_state_dict(ck["adapter"])
    for m in (base1, base2, tuned1, tuned2, adapter):
        m.eval()

    out = {"checkpoint": a.checkpoint,
           "stock_boltz2": score(base1, base2, None, groups, have, dev),
           "finetuned": score(tuned1, tuned2, adapter, groups, have, dev)}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
