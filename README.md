# POEM: a from-scratch symbolic melody generator

POEM trains small CPU-friendly autoregressive symbolic melody models on the local Beautiful-Motifs MIDI dataset. The primary run uses short motifs only; pass `--include_long_motifs` to include the long folder.

## Setup

```bash
pip install -r requirements.txt
```

The code uses `pretty_midi` for note-level parsing because the tokenizer needs direct note start/end times, velocities, tempo estimates, and MIDI writing for generated samples.

## Dataset Discovery And EDA

```bash
python scripts/inspect_dataset.py --data_dir Beautiful-Motifs-CC-BY-NC-SA
python scripts/eda.py --data_dir Beautiful-Motifs-CC-BY-NC-SA --token_sample_size 1000
```

The tokenizer extracts a monophonic melody line from stacked notes by grouping quantized onsets, keeping the highest pitch at each onset, and clipping overlaps. Duration buckets are 32 log-spaced beat buckets from 1/32 beat to 8 beats; `scripts/eda.py` prints the exact edges.

## Tests

```bash
python -m pytest -q
```

Important focused tests:

```bash
python -m pytest tests/test_tokenizer_roundtrip.py -q
python -m pytest tests/test_gated_deltanet.py -q
python -m pytest tests/test_act_halting.py -q
```

## Smoke Tests

Smoke mode uses 50 train clips, 10 validation clips, tiny model dimensions, and short step budgets.

```bash
python train.py --model_type D --data_dir Beautiful-Motifs-CC-BY-NC-SA --smoke_test --max_steps 60
python train.py --model_type C --data_dir Beautiful-Motifs-CC-BY-NC-SA --smoke_test --max_steps 60
python train.py --model_type B --data_dir Beautiful-Motifs-CC-BY-NC-SA --smoke_test --max_steps 40
python train.py --model_type E --data_dir Beautiful-Motifs-CC-BY-NC-SA --smoke_test --max_steps 60
python train.py --model_type A --data_dir Beautiful-Motifs-CC-BY-NC-SA --smoke_test --max_steps 40
python train.py --model_type F --data_dir Beautiful-Motifs-CC-BY-NC-SA --smoke_test --max_steps 40
```

Candidate A logs `loops/token`. Full runs use an ACT threshold of 0.99 and a hard `max_loops=6`; smoke mode lowers the threshold to verify early-stop plumbing with only two loops.

Candidate F is the speed-focused hybrid recursive architecture. Each mixer splits
channels 3:1 between a Gated DeltaNet branch and dense RoPE attention branch,
then fuses them in one residual block. On Kaggle it uses the optional
`flash-linear-attention` GDN layer when installed, with POEM's sequential GDN as
a local fallback for tests and CPU debugging. F disables FLA's optional short
convolution by default on Kaggle because the core GDN path is the important
speedup and the short-conv Triton autotuner has been brittle on dual T4 sessions.

## Full Training

Full comparison runs use the same explicit epoch count for every candidate. The
default is 40 epochs, chosen to put the largest ~10.9M-parameter candidates just
over the ~20 tokens/parameter Chinchilla-style reference using the short-motif
training split.

Build the short-motif token cache once:

```bash
python -u scripts/pretokenize.py --data_dir Beautiful-Motifs-CC-BY-NC-SA --output cache/poem-short-token-cache.pt
```

```bash
python -u train.py --model_type D --data_dir Beautiful-Motifs-CC-BY-NC-SA --epochs 40 --batch_size 32 --val_interval 2000 --token_cache cache/poem-short-token-cache.pt
```

Candidate E keeps Candidate B's four-block macro depth but uses a wider attention+FFN stack so the no-GDN ablation remains near the other candidates' parameter counts.

Optional augmentations are off by default:

```bash
python train.py --model_type C --data_dir Beautiful-Motifs-CC-BY-NC-SA --pitch_transpose_aug --tempo_aug
```

Checkpoints are named `poem-a-<step>.pt` and `poem-a-best.pt` inside `checkpoints/` by default.

## Compute Budget Report

```bash
python scripts/compute_budget_report.py --data_dir Beautiful-Motifs-CC-BY-NC-SA --epochs 40
```

This reuses `training/compute_budget.py`, which reports tokens per parameter per epoch and selected total tokens per parameter against the ~20 tokens/parameter Chinchilla reference.

## Generate Samples

```bash
python generate.py --checkpoint checkpoints/poem-a-best.pt --num_samples 8 --output_dir samples
```

## Kaggle Dual-T4 Workflow

Use [kaggle_poem_dual_t4.ipynb](kaggle_poem_dual_t4.ipynb) with two Kaggle datasets attached:

- `POEM-BASE`: a mirror of this repository
- `Beautiful-Motifs-CC-BY-NC-SA`: the MIDI dataset

The notebook asks for a Hugging Face write token, creates/updates a private model repository such as `your-name/POEM-BASE`, pretokenizes the short motifs, skips already-finished D/C/E by default, trains candidates in order `F B A`, writes checkpoint-level JSON metrics plus summary JSON files, generates five MIDI samples per completed candidate, and uploads each completed model folder in a single Hugging Face commit.

Default Kaggle batch sizes are per-architecture: C/E use 256, F uses 128, B uses 64, and A uses 32. F installs `flash-linear-attention[cuda]` in the notebook so its GDN branch can use chunked GPU kernels instead of the slow sequential reference path.
