# Experiment log

Append one section per meaningful run. Do not overwrite an earlier result merely because a later run is better; reproducible failures are useful evidence too.

## Run template

- Run ID:
- Date/time and timezone:
- Part/task:
- Goal or hypothesis:
- Git commit:
- Entry command/notebook:
- Dataset and split manifest:
- Random seed(s):
- Device and host:
- Python/framework/CUDA versions:
- Precision/dtype:
- Model/configuration:
- Optimiser/schedule:
- Batch size / epochs / steps:
- Timing boundary and synchronisation method:
- Primary metrics:
- Saved artifacts:
- Result/interpretation:
- Problems or next change:

## Part 1 benchmark table

| Run | N | Method | Device | Dtype | Warm-up | Repeats | Transfer included? | Median (ms) | Spread | Correct vs FFT? |
|---|---:|---|---|---|---:|---:|---|---:|---:|---|
| | | | | | | | | | | |

### Local CPU validation — 2026-08-27 16:24 AEST

- Command: `MPLBACKEND=Agg .venv/bin/python part1.py`
- Host: Apple Silicon CPU, macOS 26.6.2; CUDA unavailable
- Environment: Python 3.11.15, NumPy 2.4.6, PyTorch 2.13.0
- Dtype: `float64` input and `complex128` DFT
- Correctness: NumPy naive DFT, PyTorch CPU naive DFT, and NumPy FFT all agreed (`allclose=True`)
- At `N=2048`: NumPy FFT ≈ 0.00001 s, PyTorch CPU naive DFT ≈ 0.014 s, NumPy naive DFT ≈ 1.01 s
- Plot check: Fourier reconstruction, odd-harmonic spectrum, and timing curve rendered correctly
- Remaining work: run the CUDA branch on Rangpur and record its timing

## Model result summary

| Run | Task | Model | Test metric | Time | Hardware | Checkpoint | Notes |
|---|---|---|---:|---:|---|---|---|
| | | | | | | | |

## Demonstration evidence checklist

- [ ] Selected figures copied to `results/figures/` with descriptive filenames.
- [ ] Compact metrics/config copied to `results/metrics/`.
- [ ] Best checkpoint path recorded (checkpoint itself remains ignored).
- [ ] Exact inference command tested from a fresh process.
- [ ] Exact one-epoch Rangpur command tested from a fresh allocation.
- [ ] Every reported number is tied to a Git commit and environment.
- [ ] Result interpretation and limitations can be explained without reading generated text.
