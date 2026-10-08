#!/usr/bin/env python3
"""Measurements, blob discovery, splits, and the comparison groups the ranking loss uses."""
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np


def load_measurements(paths):
    """One JSON object per line. Required keys: record_id, value. Anything else is metadata
    you can group or split on."""
    if isinstance(paths, str):
        paths = [paths]
    rows = []
    for p in paths:
        with open(p) as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{p}:{i}: not valid JSON ({e})")
                for k in ("record_id", "value"):
                    if k not in r:
                        raise SystemExit(f"{p}:{i}: missing required key {k!r}")
                rows.append(r)
    if not rows:
        raise SystemExit(f"no measurements found in {paths}")
    return rows


def find_blobs(blob_dirs):
    """{record_id: path}. A blob MUST be named exactly <record_id>.pt -- see README, 'Gotchas'."""
    if isinstance(blob_dirs, str):
        blob_dirs = [blob_dirs]
    idx = {}
    for d in blob_dirs:
        pattern = d if d.endswith(".pt") else os.path.join(d, "*.pt")
        for p in glob.glob(pattern):
            idx[os.path.basename(p)[:-3]] = p
    return idx


def find_extra_samples(sample_dirs, primary):
    """{record_id: [primary, extra, ...]} for the consistency loss.

    Extra Boltz-2 samples of the SAME record live in their own directory, each still named
    <record_id>.pt. They are never renamed -- see README, 'Gotchas'.
    """
    if isinstance(sample_dirs, str):
        sample_dirs = [sample_dirs]
    extra = defaultdict(list)
    prim_abs = {r: os.path.abspath(p) for r, p in primary.items()}
    for d in sample_dirs:
        pattern = d if d.endswith(".pt") else os.path.join(d, "*.pt")
        for p in sorted(glob.glob(pattern)):
            rid = os.path.basename(p)[:-3]
            if rid not in primary:
                continue
            ap = os.path.abspath(p)
            if ap != prim_abs[rid] and ap not in extra[rid]:
                extra[rid].append(ap)
    return {r: [primary[r]] + v for r, v in extra.items() if v}


def read_record_ids(src):
    """A pinned validation set: a list of record_ids, or a path to a file with one per line."""
    if src is None:
        return None
    if isinstance(src, (list, tuple, set)):
        return set(src)
    with open(src) as fh:
        return {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}


def split_records(rows, frac=0.1, by=None, seed=0, val_ids=None):
    """Hold out records for validation.

    val_ids      an explicit set of record_ids; everything else trains. Overrides frac/by.
    by=None      hold out random record_ids
    by="<key>"   hold out whole groups of that key (e.g. by="target" keeps a protein entirely
                 unseen, which is the honest test of generalisation to a new target)
    """
    if val_ids is not None:
        val_ids = set(val_ids)
        return ([r for r in rows if r["record_id"] not in val_ids],
                [r for r in rows if r["record_id"] in val_ids])
    rng = np.random.default_rng(seed)
    if by is None:
        ids = sorted({r["record_id"] for r in rows})
        rng.shuffle(ids)
        val = set(ids[:max(1, int(len(ids) * frac))])
        return ([r for r in rows if r["record_id"] not in val],
                [r for r in rows if r["record_id"] in val])
    keys = sorted({r[by] for r in rows if by in r})
    rng.shuffle(keys)
    val = set(keys[:max(1, int(len(keys) * frac))])
    return ([r for r in rows if r.get(by) not in val],
            [r for r in rows if r.get(by) in val])


def build_groups(rows, have, specs, min_delta=0.3):
    """Comparison groups for the ranking loss.

    A spec is {"group_by": [...], "vary": "<key>"}: hold group_by fixed, rank records that differ
    in `vary`. "rank mutants of one protein against one ligand" is group_by ["ligand","assay"]
    with vary "target"; swap them to rank ligands against one mutant. Any columns work.

    A group is kept only if some pair differs in `vary` by at least `min_delta` in value, so the
    loss never spends gradient on pairs the assay cannot distinguish.
    """
    rows = [r for r in rows if r["record_id"] in have]
    groups = []
    for spec in specs:
        gb, vary = spec["group_by"], spec["vary"]
        bucket = defaultdict(lambda: defaultdict(list))
        varyval = defaultdict(dict)
        for r in rows:
            if vary not in r or any(g not in r for g in gb):
                continue
            key = tuple(r[g] for g in gb)
            bucket[key][r["record_id"]].append(float(r["value"]))
            varyval[key][r["record_id"]] = r[vary]
        for key, ridmap in bucket.items():
            pts = {rid: float(np.median(v)) for rid, v in ridmap.items()}
            if len(pts) < 2:
                continue
            rids = list(pts)
            if any(varyval[key][a] != varyval[key][b] and abs(pts[a] - pts[b]) >= min_delta
                   for a, b in itertools.combinations(rids, 2)):
                groups.append({"spec": spec.get("name", f"{'+'.join(gb)}->{vary}"),
                               "key": list(key), "points": pts})
    return groups


def absolute_targets(rows, have):
    """One target per record for the absolute loss: the median of its measurements."""
    by = defaultdict(list)
    for r in rows:
        if r["record_id"] in have and not r.get("ranking_only"):
            by[r["record_id"]].append(float(r["value"]))
    return {rid: float(np.median(v)) for rid, v in by.items()}


def balance_weights(groups, rows, specs, max_weight=None):
    """Sampling weight per group, so over-represented classes stop dominating.

    specs is a list of {"by": "<column>", "alpha": 0..1, "max": <cap>}. A group's weight for one
    spec is count(class)**-alpha, normalised to mean 1. alpha=0 is uniform, alpha=1 is full
    inverse-frequency. Weights from several specs multiply, so you can balance by target and by
    ligand at once.
    """
    if not specs:
        return [1.0] * len(groups)
    rid_class = {}
    for r in rows:
        rid_class.setdefault(r["record_id"], r)
    w = np.ones(len(groups), dtype=float)
    for spec in specs:
        col, alpha = spec["by"], float(spec.get("alpha", 1.0))
        counts = defaultdict(int)
        for r in rows:
            if col in r:
                counts[r[col]] += 1
        if not counts:
            continue
        per = []
        for g in groups:
            vals = [rid_class[r].get(col) for r in g["points"] if r in rid_class]
            vals = [v for v in vals if v is not None]
            if not vals:
                per.append(1.0); continue
            per.append(float(np.mean([counts[v] ** -alpha for v in vals])))
        per = np.asarray(per, dtype=float)
        per = per / (per.mean() or 1.0)
        cap = spec.get("max")
        if cap:
            per = np.minimum(per, float(cap))     # per-spec cap, in "x uniform" units
        w *= per
    # Normalise to mean 1 first, so a cap is expressed in "x uniform" and actually bounds the
    # weight a caller sees.
    w = w / (w.mean() or 1.0)
    if max_weight:
        w = np.minimum(w, float(max_weight))
    return w.tolist()


def cliff_weights(groups, alpha=1.0):
    """Weight groups by how much their values actually spread.

    A group whose members all measure the same is uninformative for ranking; one spanning orders
    of magnitude carries the activity cliffs you care about. weight = spread**alpha, mean 1.
    """
    if not alpha:
        return [1.0] * len(groups)
    spread = np.array([max(g["points"].values()) - min(g["points"].values()) for g in groups],
                      dtype=float)
    w = np.power(np.maximum(spread, 1e-6), float(alpha))
    return (w / (w.mean() or 1.0)).tolist()
