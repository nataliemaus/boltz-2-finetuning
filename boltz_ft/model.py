#!/usr/bin/env python3
"""Affinity heads, the residual adapter, and the molecular-weight correction."""
import torch
import torch.nn as nn

TOKEN_S, TOKEN_Z = 384, 128
MW_MODEL_COEF, MW_COEF, MW_BIAS = 1.03525938, -0.59992683, 2.83288489


def load_affinity_heads(ckpt_path, device):
    """The two affinity heads from a stock Boltz-2 affinity checkpoint."""
    from boltz.model.modules.affinity import AffinityModule
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp, sd = ck["hyper_parameters"], ck["state_dict"]
    ts, tz = hp.get("token_s", TOKEN_S), hp.get("token_z", TOKEN_Z)
    out = []
    for tag, argkey in (("affinity_module1", "affinity_model_args1"),
                        ("affinity_module2", "affinity_model_args2")):
        m = AffinityModule(ts, tz, **hp[argkey])
        sub = {k[len(tag) + 1:]: v for k, v in sd.items() if k.startswith(tag + ".")}
        missing, _ = m.load_state_dict(sub, strict=False)
        assert not [k for k in missing if "boundaries" not in k], f"{tag} missing {missing}"
        out.append(m.to(device))
    return out[0], out[1]


class Adapter(nn.Module):
    """Residual MLP on the per-token representation, applied before the affinity heads.

    Zero-initialised output layer, so an untrained adapter is the identity and the model starts
    exactly at stock Boltz-2.
    """

    def __init__(self, d_s=TOKEN_S, hidden=512):
        super().__init__()
        self.ln = nn.LayerNorm(d_s)
        self.fc1 = nn.Linear(d_s, hidden)
        self.fc2 = nn.Linear(hidden, d_s)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, s):
        return s + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln(s))))


def make_adapter(device, hidden=512, d_s=TOKEN_S):
    """d_s is the per-token width. Read it off a blob rather than assuming, so a Boltz build
    with a different token_s still works: adapter_width(blob)."""
    return Adapter(d_s=d_s, hidden=hidden).to(device)


def adapter_width(blob):
    return int(blob["s_inputs"].shape[-1])


def predict_value(m1, m2, blob, device, adapter=None, use_kernels=False):
    """Average the two heads and apply Boltz-2's molecular-weight correction.

    Returns the corrected scalar in the same units as the base model's affinity output
    (log10 IC50 in uM for the stock checkpoint: lower = stronger binding).
    """
    s = blob["s_inputs"].to(device)
    z = blob["z"].to(device)
    x = blob["x_pred"].to(device)
    feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in blob["feats"].items()}
    if adapter is not None:
        s = adapter(s)
    o1 = m1(s_inputs=s, z=z, x_pred=x, feats=feats, multiplicity=1, use_kernels=use_kernels)
    o2 = m2(s_inputs=s, z=z, x_pred=x, feats=feats, multiplicity=1, use_kernels=use_kernels)
    v = (o1["affinity_pred_value"].squeeze() + o2["affinity_pred_value"].squeeze()) / 2.0
    mw = feats["affinity_mw"]
    if not torch.is_tensor(mw):
        mw = torch.as_tensor(mw, device=device)
    mw = mw.to(device).squeeze().float()
    return MW_MODEL_COEF * v + MW_COEF * (mw ** 0.3) + MW_BIAS


def trainable_parameters(m1, m2, adapter, train_heads=True):
    ps = list(adapter.parameters()) if adapter is not None else []
    if train_heads:
        ps += list(m1.parameters()) + list(m2.parameters())
    return ps
