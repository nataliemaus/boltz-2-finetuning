#!/usr/bin/env python3
"""The three objectives: absolute fit, pairwise ranking, and cross-sample consistency."""
import itertools

import torch
import torch.nn.functional as Fn


def huber(x, y, delta=1.0):
    d = x - y
    a = d.abs()
    return torch.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta))


def absolute_loss(preds, targets):
    """Fit the measured value itself. Keep its weight small: it pins the scale, but on most
    function data the ranking is what you actually care about and the absolute calibration of a
    borrowed head is not meaningful."""
    terms = [huber(p, torch.as_tensor(float(targets[r]), device=p.device))
             for r, p in preds.items() if r in targets]
    return torch.stack(terms).mean() if terms else None


def pairwise_rank_loss(group, preds, min_delta=0.3):
    """Match the predicted difference to the measured difference, within one comparison group.

    Pairs closer than min_delta are skipped: below assay resolution they are noise, and training
    on them teaches the model to reproduce that noise.
    """
    pts = group["points"]
    rids = [r for r in pts if r in preds]
    total, n = 0.0, 0.0
    for a, b in itertools.combinations(rids, 2):
        d = pts[a] - pts[b]
        if abs(d) < min_delta:
            continue
        total = total + huber(preds[a] - preds[b],
                              torch.as_tensor(float(d), device=preds[a].device))
        n += 1.0
    return total / n if n else None


def ranknet_loss(group, preds, min_delta=0.3):
    """Pairwise logistic ordering. Optimises rank-correlation directly, and unlike the Huber form
    it does not also ask the model to match the SIZE of each difference."""
    pts = group["points"]
    rids = [r for r in pts if r in preds]
    total, n = 0.0, 0.0
    for a, b in itertools.combinations(rids, 2):
        d = pts[a] - pts[b]
        if abs(d) < min_delta:
            continue
        y = torch.as_tensor(1.0 if d > 0 else 0.0, device=preds[a].device)
        total = total + Fn.binary_cross_entropy_with_logits(preds[a] - preds[b], y)
        n += 1.0
    return total / n if n else None


def consistency_loss(sample_preds, target=0.25, detach_ref=True):
    """Penalise disagreement between predictions for independent Boltz-2 samples of the SAME
    complex.

    Boltz-2's structure sampling is stochastic, so one record gives a different prediction each
    pass. Spread across samples is therefore pure prediction noise, and shrinking it makes a
    single pass more informative. `target` leaves a floor so the model is not pushed toward
    ignoring the structure altogether.
    """
    if len(sample_preds) < 2:
        return None
    stacked = torch.stack(sample_preds)
    ref = stacked.mean()
    if detach_ref:
        ref = ref.detach()
    spread = (stacked - ref).abs().mean()
    return Fn.relu(spread - target)


def concordance(pred, meas):
    """Fraction of correctly ordered pairs; 0.5 is chance."""
    good = tot = 0.0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if meas[i] == meas[j]:
                continue
            tot += 1
            d = (pred[i] - pred[j]) * (meas[i] - meas[j])
            good += 1.0 if d > 0 else (0.5 if pred[i] == pred[j] else 0.0)
    return good / tot if tot else float("nan")
