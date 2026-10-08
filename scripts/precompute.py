#!/usr/bin/env python3
"""Step 1: build one blob per record.

    python scripts/precompute.py --yaml-dir yamls/ --out blobs/

Runs Boltz-2 once per YAML and caches what the affinity head consumes. Skips records whose blob
already exists, so it is safe to re-run after an interruption. Use --sample-index to build a
second, independent set of samples for the consistency loss; that writes to <out>/sample<N>/.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_BLOB_BYTES = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-dir", required=True, help="directory of Boltz-2 input YAMLs, one per record")
    ap.add_argument("--out", required=True, help="directory to write <record_id>.pt into")
    ap.add_argument("--sample-index", type=int, default=0,
                    help="N>0 writes to <out>/sample<N>/ for the consistency loss")
    ap.add_argument("--boltz-cache", default=None, help="passed to boltz as --cache")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed through to `boltz predict`")
    a = ap.parse_args()

    out = Path(a.out) if a.sample_index == 0 else Path(a.out) / f"sample{a.sample_index}"
    out.mkdir(parents=True, exist_ok=True)
    yamls = sorted(Path(a.yaml_dir).glob("*.yaml")) + sorted(Path(a.yaml_dir).glob("*.yml"))
    if not yamls:
        sys.exit(f"no .yaml files in {a.yaml_dir}")

    done = skipped = failed = 0
    for i, y in enumerate(yamls, 1):
        # The blob filename IS the record id. Keep the YAML stem equal to the record id.
        blob = out / f"{y.stem}.pt"
        if blob.exists() and blob.stat().st_size > MIN_BLOB_BYTES:
            skipped += 1
            continue
        if blob.exists():
            blob.unlink()        # truncated from an earlier interruption
        cmd = [sys.executable, str(HERE / "affinity_hook.py"), str(y)]
        if a.boltz_cache:
            cmd += ["--cache", a.boltz_cache]
        cmd += a.extra
        env = dict(os.environ, BLOB=str(blob))
        r = subprocess.run(cmd, env=env)
        ok = r.returncode == 0 and blob.exists() and blob.stat().st_size > MIN_BLOB_BYTES
        done += ok
        failed += (not ok)
        print(f"[precompute] {i}/{len(yamls)} {y.stem} {'ok' if ok else 'FAILED'}", flush=True)

    print(f"[precompute] {done} written, {skipped} already present, {failed} failed -> {out}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
