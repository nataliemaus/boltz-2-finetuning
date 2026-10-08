#!/usr/bin/env python3
"""Run the whole training loop on synthetic blobs, with no GPU and no Boltz-2 installed.

    python scripts/demo.py

This exists to check your install and to show you what a run looks like before you spend hours
precomputing real blobs. The "model" here is a randomly initialised stand-in, NOT Boltz-2, so the
numbers mean nothing. Everything else -- data loading, grouping, splitting, the losses, the loop,
checkpointing -- is the same code the real pipeline runs.
"""
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import boltz_ft.model as model_mod

D_S = 32


class _StandInHead(torch.nn.Module):
    """Stands in for one Boltz-2 affinity head so the demo needs no checkpoint."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.w = torch.nn.Linear(D_S, 1)
        with torch.no_grad():
            self.w.weight.mul_(0.01)
            self.w.bias.zero_()

    def forward(self, s_inputs=None, z=None, x_pred=None, feats=None, **kw):
        return {"affinity_pred_value": self.w(s_inputs.mean(1)).squeeze(-1)}


def main():
    model_mod.load_affinity_heads = lambda ckpt, dev: (_StandInHead(0).to(dev),
                                                       _StandInHead(1).to(dev))
    tmp = Path(tempfile.mkdtemp(prefix="boltz_ft_demo_"))
    blobs = tmp / "blobs"
    blobs.mkdir()

    # Synthetic features that actually carry signal: one direction of s_inputs tracks the
    # value, buried in noise. Without that there is nothing to learn and the demo would show a
    # flat run -- which is correct behaviour, but a poor first example.
    rows = []
    torch.manual_seed(0)
    direction = torch.randn(D_S)
    for i, target in enumerate(f"KIN{k}" for k in range(1, 16)):
        for j, lig in enumerate(("ligand_a", "ligand_b", "ligand_c", "ligand_d")):
            rid = f"{target}__{lig}"
            value = float(i + 3.0 * j)
            torch.manual_seed(i * 100 + j)
            s = torch.randn(1, 7, D_S) + 0.06 * value * direction
            torch.save({"s_inputs": s,
                        "z": torch.randn(1, 7, 7, 4),
                        "x_pred": torch.randn(1, 7, 3),
                        "feats": {"affinity_mw": torch.tensor([300.0])}}, blobs / f"{rid}.pt")
            rows.append({"record_id": rid, "value": value,
                         "target": target, "ligand": lig, "assay": "demo"})
    meas = tmp / "measurements.jsonl"
    meas.write_text("".join(json.dumps(r) + "\n" for r in rows))

    cfg = {"run_name": "demo", "out_dir": str(tmp / "runs"),
           "measurements": [str(meas)], "blob_dirs": [str(blobs)], "affinity_ckpt": "STAND-IN",
           "ranking_groups": [{"group_by": ["ligand", "assay"], "vary": "target"},
                              {"group_by": ["target", "assay"], "vary": "ligand"}],
           "adapter_hidden": 32, "max_epochs": 6, "patience": 6,
           "val_frac": 0.3, "val_by": "target", "device": "cpu",
           "lambda_rank": 8.0, "lambda_abs": 0.25, "records_per_group": 8, "lr": 6e-4}
    cfg_path = tmp / "demo_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))

    print(f"[demo] synthetic data in {tmp}")
    print("[demo] the model is a stand-in, so the numbers below are meaningless by design\n")
    sys.argv = ["train.py", "--config", str(cfg_path)]
    import runpy
    runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
