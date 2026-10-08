#!/usr/bin/env python3
"""Capture the affinity-head inputs during a normal `boltz predict` run.

Boltz-2 builds the expensive part (trunk, structure samples) and then hands the affinity head a
small set of tensors. This patches AffinityModule.forward to write those tensors to $BLOB, then
calls through. Run it exactly like `boltz predict`.

    BLOB=/path/<record_id>.pt python scripts/affinity_hook.py input.yaml --no_kernels
"""
import os
import sys

import torch
import boltz.model.modules.affinity as affmod

if "BLOB" not in os.environ:
    sys.exit("[hook] set $BLOB to the output path, e.g. BLOB=blobs/<record_id>.pt")
BLOB = os.environ["BLOB"]

_orig = affmod.AffinityModule.forward
_state = {"done": False}


def _patched(self, *args, **kwargs):
    # The head is called TWICE per record (Boltz-2 ensembles two modules over identical inputs).
    # Save once.
    if not _state["done"]:
        names = ["s_inputs", "z", "x_pred", "feats"]
        v = {n: (kwargs[n] if n in kwargs else (args[i] if i < len(args) else None))
             for i, n in enumerate(names)}
        missing = [n for n in names if v[n] is None]
        if missing:
            raise RuntimeError(
                f"[hook] AffinityModule.forward did not supply {missing}. This patches a Boltz-2 "
                f"internal, so a different release can change the signature. Pin your version.")
        blob = {
            "s_inputs": v["s_inputs"].detach().cpu(),
            "z": v["z"].detach().cpu(),
            "x_pred": v["x_pred"].detach().cpu(),
            "feats": {k: (t.detach().cpu() if torch.is_tensor(t) else t)
                      for k, t in v["feats"].items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(BLOB)) or ".", exist_ok=True)
        # Write atomically. A job killed mid-save otherwise leaves a truncated .pt that a resume
        # counts as finished.
        torch.save(blob, BLOB + ".tmp")
        os.replace(BLOB + ".tmp", BLOB)
        _state["done"] = True
        print(f"[hook] wrote {BLOB}", flush=True)
    return _orig(self, *args, **kwargs)


affmod.AffinityModule.forward = _patched

if __name__ == "__main__":
    # Keep the argv rewrite inside the guard: Boltz's dataloader re-imports this module in
    # worker processes, where rewriting argv corrupts their arguments.
    from boltz.main import cli
    if not (len(sys.argv) > 1 and sys.argv[1] == "predict"):
        sys.argv = ["boltz", "predict"] + sys.argv[1:]
    cli()
