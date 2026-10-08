#!/usr/bin/env python3
"""Step 2: fine-tune on the cached blobs.

    python scripts/train.py --config configs/starter.json

Writes to <out_dir>/<run_name>/: a checkpoint per improving epoch, metrics.jsonl (one row per
epoch), and the resolved config. Nothing outside that directory is touched.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boltz_ft import data as D
from boltz_ft import losses as L
from boltz_ft.model import (adapter_width, load_affinity_heads, make_adapter,
                            predict_value, trainable_parameters)

DEFAULTS = {
    "blob_dirs": [], "sample_dirs": [], "measurements": [],
    "affinity_ckpt": None, "out_dir": "runs", "run_name": "run",
    "ranking_groups": [], "val_frac": 0.1, "val_by": None, "val_records": None,
    "min_delta": 0.3,
    "lambda_abs": 0.25, "lambda_rank": 8.0, "lambda_ranknet": 0.0, "lambda_consistency": 0.0,
    "consistency_target": 0.25, "consistency_records_per_step": 1,
    "adapter_hidden": 512, "train_heads": True,
    "lr": 1e-6, "max_lr": None, "warmup_steps": 0, "weight_decay": 1e-3, "grad_clip": 0.5,
    "groups_per_step": 1, "records_per_group": 8, "accum_steps": 4,
    "balance": [], "cliff_alpha": 0.0, "max_sample_weight": None,
    "max_epochs": 30, "patience": 10, "seed": 0, "device": None,
}


def resolve(cfg):
    out = dict(DEFAULTS)
    out.update(cfg)
    for k in ("blob_dirs", "sample_dirs", "measurements"):
        if isinstance(out[k], str):
            out[k] = [out[k]]
    if not out["measurements"]:
        sys.exit("config: 'measurements' is required (one or more .jsonl files)")
    if not out["blob_dirs"]:
        sys.exit("config: 'blob_dirs' is required (where precompute.py wrote the blobs)")
    if not out["affinity_ckpt"]:
        sys.exit("config: 'affinity_ckpt' is required (the stock Boltz-2 affinity checkpoint)")
    if not out["ranking_groups"] and out["lambda_rank"]:
        sys.exit("config: 'ranking_groups' is empty but lambda_rank > 0")
    return out


def evaluate(m1, m2, adapter, groups, targets, have, dev, cache, max_groups=None):
    """Mean per-group concordance plus absolute error, with no gradient."""
    m1.eval(); m2.eval()
    if adapter is not None:
        adapter.eval()
    concs, errs = [], []
    with torch.no_grad():
        for g in (groups if max_groups is None else groups[:max_groups]):
            rids = [r for r in g["points"] if r in have]
            if len(rids) < 2:
                continue
            preds = [float(predict_value(m1, m2, cache.get(r), dev, adapter)) for r in rids]
            meas = [g["points"][r] for r in rids]
            c = L.concordance(preds, meas)
            if c == c:
                concs.append(c)
            for r, p in zip(rids, preds):
                if r in targets:
                    errs.append(abs(p - targets[r]))
    m1.train(); m2.train()
    if adapter is not None:
        adapter.train()
    return (float(np.mean(concs)) if concs else float("nan"),
            float(np.mean(errs)) if errs else float("nan"))


class BlobCache:
    """Blobs are large; hold a bounded number in memory and read the rest from disk."""

    def __init__(self, paths, limit=256):
        self.paths, self.limit, self.store = paths, limit, {}

    def get(self, rid, path=None):
        p = path or self.paths[rid]
        if p in self.store:
            return self.store[p]
        b = torch.load(p, map_location="cpu", weights_only=False)
        if len(self.store) < self.limit:
            self.store[p] = b
        return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = resolve(json.load(open(a.config)))

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    dev = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(cfg["out_dir"]) / cfg["run_name"]
    run.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(run / "config.json", "w"), indent=1)

    rows = D.load_measurements(cfg["measurements"])
    have = D.find_blobs(cfg["blob_dirs"])
    samples = D.find_extra_samples(cfg["sample_dirs"], have) if cfg["sample_dirs"] else {}
    covered = {r["record_id"] for r in rows} & set(have)
    print(f"[data] {len(rows)} measurements, {len(have)} blobs, {len(covered)} records usable")
    if not covered:
        sys.exit("[data] no measurement has a matching blob. A blob must be named exactly "
                 "<record_id>.pt -- see README, 'Gotchas'.")
    if cfg["lambda_consistency"] and not samples:
        print("[data] WARNING: lambda_consistency > 0 but no record has a second sample; "
              "the term will be inactive. Build one with precompute.py --sample-index 1.")

    val_ids = D.read_record_ids(cfg["val_records"])
    if val_ids is not None:
        missing = len(val_ids - {r["record_id"] for r in rows})
        print(f"[data] validation: {len(val_ids)} pinned record(s) from {cfg['val_records']}"
              + (f", {missing} not present in the measurements" if missing else ""))
    else:
        print(f"[data] validation: random {cfg['val_frac']:.0%} held out"
              + (f" by {cfg['val_by']!r}" if cfg["val_by"] else " by record"))
    tr_rows, va_rows = D.split_records(rows, cfg["val_frac"], cfg["val_by"], cfg["seed"], val_ids)
    tr_groups = D.build_groups(tr_rows, have, cfg["ranking_groups"], cfg["min_delta"])
    va_groups = D.build_groups(va_rows, have, cfg["ranking_groups"], cfg["min_delta"])
    tr_tgt = D.absolute_targets(tr_rows, have)
    va_tgt = D.absolute_targets(va_rows, have)
    print(f"[data] train: {len(tr_groups)} groups / {len(tr_tgt)} records | "
          f"val: {len(va_groups)} groups / {len(va_tgt)} records")
    if not tr_groups:
        sys.exit("[data] no training group survived. Check 'ranking_groups' column names and "
                 "'min_delta'.")
    if not va_groups:
        # Without these, every epoch scores nan, nothing is ever saved, and the run ends looking
        # like fine-tuning did not help. Stop now rather than after the GPU time.
        sys.exit("[data] no validation group survived, so no checkpoint could ever be saved. "
                 "Raise 'val_frac', change 'val_by', or widen 'val_records'.")

    m1, m2 = load_affinity_heads(cfg["affinity_ckpt"], dev)
    cache = BlobCache(have)
    d_s = adapter_width(cache.get(sorted(covered)[0]))
    adapter = make_adapter(dev, hidden=cfg["adapter_hidden"], d_s=d_s)
    params = trainable_parameters(m1, m2, adapter, cfg["train_heads"])
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Linear warmup from lr to max_lr, then hold. Starting small matters when the heads are
    # pretrained and the adapter is not: a large first step can undo the pretrained weights
    # before the adapter has learned anything useful.
    base_lr, peak, warm = cfg["lr"], cfg["max_lr"], int(cfg["warmup_steps"])
    def set_lr(global_step):
        if not peak or not warm:
            return base_lr
        f = min(1.0, global_step / float(warm))
        lr = base_lr + f * (peak - base_lr)
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        return lr

    spec_weight = {sp.get("name", f"{'+'.join(sp['group_by'])}->{sp['vary']}"): sp["weight"]
                   for sp in cfg["ranking_groups"] if "weight" in sp}
    if spec_weight:
        print(f"[loss] per-group ranking weights: {spec_weight}")

    sample_w = np.asarray(D.balance_weights(tr_groups, tr_rows, cfg["balance"],
                                            cfg["max_sample_weight"]), dtype=float)
    sample_w = sample_w * np.asarray(D.cliff_weights(tr_groups, cfg["cliff_alpha"]), dtype=float)
    sample_w = sample_w / (sample_w.sum() or 1.0)
    if cfg["balance"] or cfg["cliff_alpha"]:
        print(f"[data] sampling weights: min {sample_w.min()*len(sample_w):.2f}x  "
              f"max {sample_w.max()*len(sample_w):.2f}x  (1.00x = uniform)")

    # The adapter is zero-initialised, so epoch 0 IS stock Boltz-2. Record it as the baseline
    # every later epoch has to beat.
    c0, e0 = evaluate(m1, m2, adapter, va_groups, va_tgt, have, dev, cache)
    print(f"[epoch 0] (untrained adapter = stock Boltz-2)  val_concordance {c0:.4f}  val_abs_err {e0:.4f}")
    best, best_epoch, metrics_path = c0, 0, run / "metrics.jsonl"
    with open(metrics_path, "w") as fh:
        fh.write(json.dumps({"epoch": 0, "val_concordance": c0, "val_abs_err": e0}) + "\n")

    rng = random.Random(cfg["seed"])
    gstep = 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        # One "epoch" is len(tr_groups) sampled groups. With weights this is sampling WITH
        # replacement, so a heavily weighted group can appear more than once.
        if cfg["balance"] or cfg["cliff_alpha"]:
            order = list(np.random.default_rng(cfg["seed"] + epoch)
                         .choice(len(tr_groups), size=len(tr_groups), replace=True, p=sample_w))
        else:
            order = list(range(len(tr_groups)))
            rng.shuffle(order)
        tot, nstep = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, gi in enumerate(order, 1):
            g = tr_groups[gi]
            rids = [r for r in g["points"] if r in have]
            rng.shuffle(rids)
            rids = rids[:cfg["records_per_group"]]
            if len(rids) < 2:
                continue
            preds = {r: predict_value(m1, m2, cache.get(r), dev, adapter) for r in rids}

            loss = torch.zeros((), device=dev)
            # a ranking_groups entry may carry its own "weight"; otherwise lambda_rank applies
            lam = spec_weight.get(g["spec"], cfg["lambda_rank"])
            if lam:
                t = L.pairwise_rank_loss(g, preds, cfg["min_delta"])
                if t is not None:
                    loss = loss + lam * t
            if cfg["lambda_ranknet"]:
                t = L.ranknet_loss(g, preds, cfg["min_delta"])
                if t is not None:
                    loss = loss + cfg["lambda_ranknet"] * t
            if cfg["lambda_abs"]:
                t = L.absolute_loss(preds, tr_tgt)
                if t is not None:
                    loss = loss + cfg["lambda_abs"] * t
            if cfg["lambda_consistency"] and samples:
                used = 0
                for r in rids:
                    if used >= cfg["consistency_records_per_step"]:
                        break
                    paths = samples.get(r)
                    if not paths or len(paths) < 2:
                        continue
                    sp = [predict_value(m1, m2, cache.get(r, p), dev, adapter) for p in paths]
                    t = L.consistency_loss(sp, cfg["consistency_target"])
                    if t is not None:
                        loss = loss + cfg["lambda_consistency"] * t
                    used += 1

            if not loss.requires_grad:
                continue
            (loss / cfg["accum_steps"]).backward()
            tot += float(loss); nstep += 1
            # groups_per_step groups share one optimiser step, on top of accum_steps
            if step % (cfg["accum_steps"] * cfg["groups_per_step"]) == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
                set_lr(gstep); gstep += 1
                opt.step(); opt.zero_grad(set_to_none=True)
        opt.step(); opt.zero_grad(set_to_none=True)

        vc, ve = evaluate(m1, m2, adapter, va_groups, va_tgt, have, dev, cache)
        row = {"epoch": epoch, "train_loss": tot / max(nstep, 1), "lr": set_lr(gstep),
               "val_concordance": vc, "val_abs_err": ve, "seconds": round(time.time() - t0, 1)}
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[epoch {epoch}] loss {row['train_loss']:.4f}  val_concordance {vc:.4f}  "
              f"val_abs_err {ve:.4f}  ({row['seconds']}s)", flush=True)

        if vc == vc and vc > best:
            best, best_epoch = vc, epoch
            torch.save({"m1": m1.state_dict(), "m2": m2.state_dict(),
                        "adapter": adapter.state_dict(),
                        "adapter_cfg": {"hidden": cfg["adapter_hidden"], "d_s": d_s}},
                       run / f"epoch_{epoch:03d}.pt")
            print(f"           saved epoch_{epoch:03d}.pt (best so far)", flush=True)
        if epoch - best_epoch >= cfg["patience"]:
            print(f"[stop] no val improvement in {cfg['patience']} epochs")
            break

    print(f"[done] best epoch {best_epoch}, val_concordance {best:.4f}  ->  {run}")
    if best_epoch == 0:
        print("[done] NOTE: no epoch beat the untrained adapter. That is a real result, not a "
              "crash -- see README, 'Reading the result'.")


if __name__ == "__main__":
    main()
