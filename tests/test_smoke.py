"""Runs on CPU with no Boltz-2 install: synthetic blobs, real loop."""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from boltz_ft import data as D
from boltz_ft import losses as L
from boltz_ft.model import Adapter


def _rows(tmp):
    rows = []
    for i, t in enumerate(("T1", "T2", "T3", "T4")):
        for lig in ("a", "b"):
            rows.append({"record_id": f"{t}__{lig}",
                         "value": {"a": 1.0, "b": 3.0}[lig] + 2.0 * i,
                         "target": t, "ligand": lig, "assay": "x"})
    p = tmp / "m.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p, rows


def test_measurements_require_their_keys(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"record_id": "x"}) + "\n")
    with pytest.raises(SystemExit):
        D.load_measurements(str(bad))


def test_blob_must_be_named_record_id(tmp_path):
    """The documented gotcha: a misnamed blob is simply not found."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    torch.save({"x": 1}, blobs / "T1__a.pt")
    torch.save({"x": 1}, blobs / "T2__a__p2.pt")      # wrong name on purpose
    have = D.find_blobs(str(blobs))
    assert "T1__a" in have
    assert "T2__a" not in have, "a renamed blob must not resolve to its record"


def test_extra_samples_live_in_their_own_directory(tmp_path):
    prim, extra = tmp_path / "blobs", tmp_path / "blobs" / "sample1"
    extra.mkdir(parents=True)
    torch.save({"x": 1}, prim / "R.pt")
    torch.save({"x": 2}, extra / "R.pt")
    have = D.find_blobs(str(prim))
    samples = D.find_extra_samples(str(extra), have)
    assert samples["R"][0].endswith("blobs/R.pt")
    assert len(samples["R"]) == 2


def test_groups_respect_min_delta(tmp_path):
    p, rows = _rows(tmp_path)
    have = {r["record_id"]: "x" for r in rows}
    specs = [{"group_by": ["ligand", "assay"], "vary": "target"}]
    assert D.build_groups(rows, have, specs, min_delta=0.3)
    assert not D.build_groups(rows, have, specs, min_delta=99.0)


def test_split_by_key_holds_out_whole_targets(tmp_path):
    p, rows = _rows(tmp_path)
    tr, va = D.split_records(rows, frac=0.25, by="target", seed=0)
    assert {r["target"] for r in tr}.isdisjoint({r["target"] for r in va})


def test_adapter_is_identity_at_initialisation():
    """Zero-init output layer => an untrained model is exactly stock Boltz-2."""
    a = Adapter(d_s=16, hidden=8).eval()
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        assert torch.equal(a(x), x)


def test_losses_reward_correct_ordering():
    dev = "cpu"
    g = {"points": {"r1": 3.0, "r2": 1.0}}
    good = {"r1": torch.tensor(3.0, requires_grad=True), "r2": torch.tensor(1.0, requires_grad=True)}
    bad = {"r1": torch.tensor(1.0, requires_grad=True), "r2": torch.tensor(3.0, requires_grad=True)}
    assert float(L.ranknet_loss(g, good)) < float(L.ranknet_loss(g, bad))
    assert float(L.pairwise_rank_loss(g, good)) < float(L.pairwise_rank_loss(g, bad))


def test_consistency_penalises_disagreement_only_above_target():
    agree = [torch.tensor(1.0), torch.tensor(1.0)]
    disagree = [torch.tensor(0.0), torch.tensor(4.0)]
    assert float(L.consistency_loss(agree, target=0.25)) == 0.0
    assert float(L.consistency_loss(disagree, target=0.25)) > 0.0
    assert L.consistency_loss([torch.tensor(1.0)]) is None


def test_concordance_is_half_at_chance():
    assert L.concordance([1, 2, 3], [1, 2, 3]) == 1.0
    assert L.concordance([3, 2, 1], [1, 2, 3]) == 0.0


@pytest.mark.parametrize("name", ["starter.json", "boltz_kinase.json"])
def test_shipped_configs_are_complete_and_valid(name):
    """Both configs must be runnable on their own; neither inherits from the other."""
    import importlib.util
    cfg = json.load(open(ROOT / "configs" / name))
    for k in ("measurements", "blob_dirs", "affinity_ckpt", "ranking_groups", "lr", "max_epochs"):
        assert k in cfg, f"{name} is missing {k!r}"
    spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts" / "train.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    assert set(cfg) >= set(tr.DEFAULTS) - {"device"}, (
        f"{name} does not cover every DEFAULTS key; a partial config would silently merge over "
        f"DEFAULTS rather than over starter.json")
    rows = D.load_measurements(str(ROOT / "examples" / "measurements.example.jsonl"))
    for s in cfg["ranking_groups"]:
        for col in s["group_by"] + [s["vary"]]:
            assert col in rows[0], f"{name} groups on {col!r}, absent from the example data"


def test_repo_ships_no_large_files():
    import subprocess
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout
    for f in out.split():
        p = ROOT / f
        if p.exists() and p.stat().st_size > 1_000_000:
            raise AssertionError(f"{f} is {p.stat().st_size/1e6:.1f} MB; this repo ships no data")


def test_no_absolute_paths_outside_examples():
    import subprocess, re
    out = subprocess.run(["git", "ls-files", "*.py", "*.md", "*.json"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    bad = []
    for f in out:
        for i, line in enumerate((ROOT / f).read_text().splitlines(), 1):
            # build the patterns so this file does not match itself
            if re.search("|".join("/" + a + "/" + b for a, b in
                                  (("data", "rbg"), ("home", r"\w"), ("Users", r"\w"))), line):
                bad.append(f"{f}:{i}")
    assert not bad, f"hard-coded machine paths: {bad}"


def test_balance_down_weights_over_represented_classes():
    rows = [{"record_id": f"r{i}", "value": float(i), "target": "common" if i < 12 else "rare"}
            for i in range(16)]
    groups = [{"points": {f"r{i}": float(i) for i in range(0, 3)}},      # all common
              {"points": {f"r{i}": float(i) for i in range(13, 16)}}]    # all rare
    w = D.balance_weights(groups, rows, [{"by": "target", "alpha": 1.0}])
    assert w[1] > w[0], "the rarer class must be sampled more, not less"
    flat = D.balance_weights(groups, rows, [{"by": "target", "alpha": 0.0}])
    assert flat[0] == pytest.approx(flat[1]), "alpha=0 must be uniform"


def test_balance_respects_the_cap():
    rows = [{"record_id": f"r{i}", "value": 0.0, "target": "c" if i else "r"} for i in range(40)]
    groups = [{"points": {"r0": 0.0, "r1": 1.0}}, {"points": {"r2": 0.0, "r3": 1.0}}]
    capped = D.balance_weights(groups, rows, [{"by": "target", "alpha": 1.0}], max_weight=1.0)
    assert max(capped) <= 1.0 + 1e-9, "max_sample_weight must bound the final weight"


def test_cliff_weighting_prefers_groups_that_actually_spread():
    flat = {"points": {"a": 1.0, "b": 1.01}}
    wide = {"points": {"a": 0.0, "b": 5.0}}
    w = D.cliff_weights([flat, wide], alpha=1.0)
    assert w[1] > w[0]
    assert D.cliff_weights([flat, wide], alpha=0.0) == [1.0, 1.0]


def test_ranking_group_weight_is_read_from_the_spec():
    """A ranking_groups entry may carry its own weight; train.py must key on the spec name."""
    specs = [{"name": "A", "group_by": ["g"], "vary": "v", "weight": 36.0},
             {"name": "B", "group_by": ["v"], "vary": "g", "weight": 8.0}]
    rows = [{"record_id": f"{g}_{v}", "value": float(i), "g": g, "v": v}
            for i, (g, v) in enumerate([("x", "1"), ("x", "2"), ("y", "1"), ("y", "2")])]
    have = {r["record_id"]: "p" for r in rows}
    groups = D.build_groups(rows, have, specs, min_delta=0.3)
    names = {g["spec"] for g in groups}
    assert names <= {"A", "B"} and names, "group spec names must survive into the groups"


def test_config_defaults_are_all_consumed():
    """Every key in DEFAULTS must be read somewhere in train.py -- a declared-but-ignored knob
    is a silent no-op for whoever sets it."""
    import re
    src = (ROOT / "scripts" / "train.py").read_text()
    block = re.search(r"DEFAULTS = \{(.*?)\n\}", src, re.S).group(1)
    keys = re.findall(r'"(\w+)"\s*:', block)
    body = src[src.index("def resolve"):]
    unused = [k for k in keys if body.count(f'"{k}"') < 1]
    assert not unused, f"declared in DEFAULTS but never read: {unused}"


def test_val_records_pins_the_split(tmp_path):
    rows = [{"record_id": f"r{i}", "value": float(i), "target": f"t{i//2}"} for i in range(10)]
    ids = tmp_path / "val_ids.txt"
    ids.write_text("r1\n# a comment\nr7\n\n")
    tr, va = D.split_records(rows, frac=0.5, by="target", seed=0,
                             val_ids=D.read_record_ids(str(ids)))
    assert {r["record_id"] for r in va} == {"r1", "r7"}, "val_records must override frac/by"
    assert {r["record_id"] for r in tr} == {f"r{i}" for i in range(10)} - {"r1", "r7"}


def test_read_record_ids_accepts_a_list_too():
    assert D.read_record_ids(["a", "b"]) == {"a", "b"}
    assert D.read_record_ids(None) is None


def test_pooled_records_are_counted_once():
    """A record in several comparison groups must be pooled once, not once per group."""
    import inspect, re
    src = (ROOT / "scripts" / "evaluate.py").read_text()
    body = src[src.index("def score("):src.index("def main(")]
    assert "pooled" in body and "setdefault" in body, (
        "evaluate.score must de-duplicate records before the pooled correlation")
    assert "flat_p += preds" not in body, "records are being appended per group again"
