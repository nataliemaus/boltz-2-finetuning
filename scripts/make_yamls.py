#!/usr/bin/env python3
"""Step 0: turn your records into one Boltz-2 input YAML each.

    python scripts/make_yamls.py --records records.jsonl --out yamls/

Each input line needs record_id, sequence and smiles:

    {"record_id": "KIN1_M1__ligand_a", "sequence": "MABC...", "smiles": "CC(=O)..."}

The YAML is named <record_id>.yaml, because precompute.py uses the stem as the record id.
Add --msa-dir to attach a precomputed alignment per record (<record_id>.a3m); without one
Boltz-2 will build its own, which is slower and needs network access.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--msa-dir", default=None)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    seen = set()
    with open(a.records) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k in ("record_id", "sequence", "smiles"):
                if k not in r:
                    sys.exit(f"{a.records}:{i}: missing {k!r}")
            rid = r["record_id"]
            if rid in seen:
                sys.exit(f"{a.records}:{i}: duplicate record_id {rid!r}; ids must be unique "
                         f"because they name the blob")
            seen.add(rid)
            msa = ""
            if a.msa_dir:
                p = Path(a.msa_dir) / f"{rid}.a3m"
                if not p.exists():
                    sys.exit(f"--msa-dir given but {p} is missing")
                msa = f"\n      msa: {p}"
            (out / f"{rid}.yaml").write_text(
                "version: 1\n"
                "sequences:\n"
                "  - protein:\n"
                "      id: A\n"
                f"      sequence: {r['sequence']}{msa}\n"
                "  - ligand:\n"
                "      id: B\n"
                f"      smiles: '{r['smiles']}'\n"
                "properties:\n"
                "  - affinity:\n"
                "      binder: B\n")
            n += 1
    print(f"[make_yamls] wrote {n} YAML(s) -> {out}")


if __name__ == "__main__":
    main()
