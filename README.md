# Fine-tuning Boltz-2 on function data

Specialise Boltz-2's affinity prediction for your own numeric readout — IC50, Kd, a selectivity
index, a growth assay — by training a small head on top of a frozen Boltz-2. Kinase inhibitor
IC50 is the running example; nothing here is kinase-specific.

## The idea

Boltz-2 spends almost all its compute folding the protein and sampling structures. The affinity
head that turns that into a number is tiny and reads only a handful of tensors. So:

1. Run Boltz-2 **once** per (protein, ligand) pair and cache those tensors — one cached file is a **blob**.
2. Train on the blobs. Each step is a forward pass through a small head, so epochs take minutes.

The expensive part happens once; you can then try many objectives and hyperparameters against the
same cache.

### What a blob is

One `.pt` file per record, holding exactly what `AffinityModule.forward` receives:

| key | what it is |
|---|---|
| `s_inputs` | per-token representation, already detached |
| `z` | pair representation |
| `x_pred` | the best sampled structure, chosen by ipTM |
| `feats` | token masks, `mol_type`, `affinity_mw` |

Boltz-2 applies a molecular-weight correction *after* the head, so predictions are only
comparable to measurements once it is applied. `boltz_ft/model.py` does this.

Blobs are large: hundreds of MB per thousand records. Budget disk first.

## Steps

```bash
pip install -r requirements.txt
pip install torch boltz==2.2.1        # steps 1 and 3 run Boltz-2 itself

python scripts/make_yamls.py  --records records.jsonl --out yamls/
python scripts/precompute.py  --yaml-dir yamls/ --out blobs/ --extra --no_kernels
python scripts/train.py       --config configs/starter.json
python scripts/evaluate.py    --config configs/starter.json --checkpoint runs/starter/epoch_007.pt
```

Anything after `--extra` is passed to `boltz predict`; use `--no_kernels` unless you installed
the optional cuequivariance kernels. Precompute is the slow step — roughly a GPU-minute per
record — and resumes where it stopped if interrupted.

Nothing is hard-coded: you choose where data, blobs and runs live.

`evaluate.py` prints stock Boltz-2 and your checkpoint on the same held-out records:

```json
{"stock_boltz2": {"concordance": 0.58, "spearman": 0.31},
 "finetuned":    {"concordance": 0.66, "spearman": 0.44}}
```

## Quick example run

Checks your install before you commit GPU time. No GPU, no Boltz-2: synthetic blobs, a stand-in
head, the real loop.

```bash
python scripts/demo.py
```

```
[data] 60 measurements, 60 blobs, 60 records usable
[data] train: 15 groups / 44 records | val: 8 groups / 16 records
[epoch 0] (untrained adapter = stock Boltz-2)  val_concordance 0.1875  val_abs_err 12.4954
[epoch 1] loss 41.0491  val_concordance 0.9167  val_abs_err 12.4521  (0.1s)
           saved epoch_001.pt (best so far)
[epoch 2] loss 39.7809  val_concordance 0.9375  val_abs_err 12.4236  (0.0s)
           saved epoch_002.pt (best so far)
...
[done] best epoch 2, val_concordance 0.9375
```

The numbers are meaningless. What matters is the shape: a usable-record count matching your rows,
groups on both sides of the split, epoch 0 as the baseline, checkpoints only on improvement.

**If `records usable` is 0 or far below your row count, stop.** That is the blob-naming problem
in *Gotchas*; training will not fix it.

## Your data

One JSON object per line. `record_id` and `value` are required; other keys are yours to group and
split on.

```json
{"record_id": "KIN1_M1__ligand_a", "value": 5.42, "target": "KIN1_M1", "ligand": "ligand_a", "assay": "biochemical"}
```

`make_yamls.py` builds the Boltz-2 inputs from a second file carrying `record_id`, `sequence` and
`smiles`. Pass `--msa-dir` if you already have alignments; it is much faster than letting Boltz-2
build them.

### Comparison groups

A group spec holds some columns fixed and ranks records differing in another:

```json
{"group_by": ["ligand", "assay"], "vary": "target"}
```

*"for one ligand in one assay, rank the targets"*. Swap them to rank ligands against one target.
Use your own column names.

## The objectives

| term | what it asks for |
|---|---|
| `lambda_rank` | predicted difference matches measured difference, within a group |
| `lambda_ranknet` | correct ordering only, ignoring gap size |
| `lambda_abs` | the predicted value matches the measurement |
| `lambda_consistency` | independent Boltz-2 samples of one complex agree |

A `ranking_groups` entry may carry its own `weight`, overriding `lambda_rank` for that group.

Start with ranking. The absolute scale of a borrowed head is rarely meaningful, so `lambda_abs`
is there to stop the scale drifting, not to fit the data.

`lambda_consistency` needs a second blob set from `--sample-index 1`. Boltz-2's sampling is
stochastic, so spread across samples is prediction noise; shrinking it makes a single pass more
informative.

## Two starting configs

`configs/starter.json` is where to begin on a new task: ranking only, mild target balancing, and
a split that holds out whole targets. Both files are complete and runnable — neither inherits
from the other.

`configs/boltz_kinase.json` is what the Boltz-kinase checkpoint was actually trained with, more
tailored to that specific task.

## Sampling and schedule

**`balance`** reweights sampling by any column. `[{"by": "target", "alpha": 1.0}]` makes a group's
weight inverse to its class frequency, so common targets stop dominating. `alpha` 0 is uniform,
1 is full inverse-frequency; `max` caps one spec, `max_sample_weight` caps the product. Specs
multiply.

**`cliff_alpha`** weights groups by value spread: a group whose members all measure the same
teaches ranking nothing. With either knob set, an epoch samples with replacement.

**`max_lr` / `warmup_steps`** ramp the learning rate from `lr`, then hold. The heads are
pretrained and the adapter is not, so a large first step can damage the heads before the adapter
has learned anything.

## Reading the result

Epoch 0 is printed before training: the adapter's output layer is zero-initialised, so an
untrained adapter is the identity and **that row is stock Boltz-2**. Checkpoints are saved only
when validation concordance beats it.

If no epoch beats epoch 0, fine-tuning did not help on your data — a result worth having, and
cheap once the blobs are cached.

`val_by: null` holds out random records, which leaks: another measurement of the same protein
usually stays in training. `val_by: "target"` holds out whole proteins.

To pin a split instead, set `val_records` to a file with one `record_id` per line. Those records
become validation and everything else trains; `val_frac` and `val_by` are ignored. The file is
just names, so it is small enough to share alongside a paper even when the measurements are not
distributable.

## Gotchas

Each of these cost us real time.

**A blob must be named exactly `<record_id>.pt`.** The loader maps filename stem to record id; a
mismatch drops the row silently. `train.py` prints how many records matched — check it.

**Extra samples go in their own directory, never a renamed file.** `blobs/` and `blobs/sample1/`,
both containing `<record_id>.pt`. Renaming to `<record_id>__p2.pt` breaks the stem contract.
Pointing a second glob at the same directory is worse: the index overwrites, so you train on one
sample believing you have two.

**Write blobs atomically.** `precompute.py` writes `.pt.tmp` then renames; a job killed mid-save
otherwise leaves a truncated file that a resume counts as finished. Anything under 1 KB is
treated as absent and rebuilt.

**The affinity head is called twice per record.** Boltz-2 ensembles two modules over identical
inputs; the hook saves the first call. One blob trains both heads.

**Pin your Boltz-2 version.** The hook patches `AffinityModule.forward`. A different release can
change that signature — it fails loudly, but you still have to re-cache.

**Skip pairs below assay resolution.** `min_delta` drops pairs the assay cannot distinguish.
Without it, much of the gradient teaches the model to reproduce noise.

## Layout

```
boltz_ft/      model (heads + adapter + MW correction), data, losses
scripts/       make_yamls, precompute, train, evaluate, affinity_hook, demo (no-GPU smoke run)
configs/       starter.json, boltz_kinase.json
examples/      the measurement format
tests/         training loop on synthetic blobs, no GPU needed
```

No weights, blobs or datasets here. Bring your own Boltz-2 affinity checkpoint and measurements.
