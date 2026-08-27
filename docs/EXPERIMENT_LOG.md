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
