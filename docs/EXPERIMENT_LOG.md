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

### Part 4 OASIS data and VAE — 2026-09-10 AEST

- Downloaded the exact 257.5 MiB `keras_png_slices_data.zip` linked by the lab. ZIP CRC and image/mask filename pairing passed. All masks are 256×256 with values 0/85/170/255.
- Preserved course splits: train 9,664 slices / 302 cases; validation 1,120 / 35; test 544 / 17. Case IDs do not overlap across these directories.
- VAE: `OMP_NUM_THREADS=4 .venv/bin/python src/comp3710_lab2/part4.py`; 30 CPU epochs, 159.79 seconds for training and validation, validation selected epoch 26. Validation BCE+KL/image 1075.211; test BCE+KL/image 1088.193. These are objective values, not accuracy percentages.
- `--evaluate` loaded the saved model in a fresh process, regenerated test reconstructions/manifold, and reported 1088.156 test loss/image. Small differences arise from VAE posterior sampling in evaluation.
- Saved `results/part4.pt`, `results/part4_reconstructions.png`, `results/part4_manifold.png`, and VAE logs/summary under `results/metrics/`. Images visually checked: brain structure is reconstructed with smoothing; the 2D prior grid shows changing brain shape/appearance.
- UNet and GAN both passed real-OASIS mini-batch backward/parameter-update checks and checkpoint reload consistency. These checks do not establish their final assessment results.

### Rangpur access and submitted experiments — 2026-09-10 AEST

- Confirmed course OASIS at `/home/groups/comp3710/OASIS/`; login host resolves to `login0.compute.eait.uq.edu.au` and is not a GPU worker. Use `/opt/slurm/bin/` when non-login SSH does not load SLURM into PATH.
- A100 test job 586513: submitted/started 15:53:57, completed in 6 seconds. Actual CUDA tensor arithmetic passed on NVIDIA A100-PCIE-40GB, PyTorch 2.13.0+cu130. `a100-test` QoS limits each job to 20 minutes. Earlier `comp3710` probe 586507 was queued, then cancelled before it ran; this is not a current queue-time guarantee.
- User explicitly confirmed ownership of `s4983489` and authorized uploading scripts and running/saving Part 4 GPU experiments in `/home/Student/s4983489/comp3710_lab2_20260910/`.
- UNet job 586521: `a100-test`, one A100, four CPUs, 20-minute limit, 12 epochs, original course data, `results/part4_2.pt`; logs in `runs/unet-586521.log`.
- Job 586524 follows UNet completion: isolated Part 1 CUDA timing into `runs/part1_gpu.log`, then 30 GAN epochs with `results/part4_3.pt`; GAN log `runs/gan-586524.log`. Final outcomes will be recorded after completion, not inferred from submission.

### Rangpur measured results — 2026-09-10 AEST

- UNet job 586521 completed all 12 epochs. The validation criterion selected epoch 3 before evaluating the held-out 544 test images. Pixel-aggregated test DSC for original labels 0/85/170/255: **0.998847 / 0.937156 / 0.947372 / 0.971544**, all strictly greater than 0.9. Saved model and segmentation examples were downloaded to `results/part4_2.pt` and `results/part4_2.png`; the image was visually checked against the ground truth. Full log: `results/metrics/unet-586521.log`.
- Fresh-process local CPU `part4_2.py --evaluate` then loaded the trained checkpoint and evaluated the same 544 images: **0.998847 / 0.937155 / 0.947369 / 0.971544**, again all >0.9; tiny differences from CUDA AMP are numerical. Log: `results/metrics/part4_unet_local_inference.log`.
- Part 1 ran on the same A100 allocation after UNet ended and before GAN began, avoiding concurrent GPU training during timing. GPU DFT matched NumPy FFT. At N=2048, NumPy FFT **0.000031 s**, PyTorch explicit CUDA DFT **0.000498 s**, NumPy naive DFT **4.108541 s**. Results for all four N values are retained in `results/metrics/part1_gpu.log`; these are observed averages under the existing script's timing protocol, not medians or general guarantees.
- GAN job 586524 completed 30 epochs / 4,530 mini-batches on all 9,664 course training slices. Reported training time **283.34 seconds**; all recorded G/D losses are finite. Model: `results/part4_3.pt`; raw history: `results/part4_3.json`; full log: `results/metrics/gan-586524.log`; image evidence: `results/figures/part4_3/`. Inspected epoch 10, final GPU samples, the loss curve and fresh local CPU checkpoint-generated samples: brain contours, ventricles and tissue patterns visibly vary. This qualitative check does not establish a numerical realism score or an exhaustive absence-of-collapse proof; the demonstrator judges image realism.
- Fresh-process `part4_3.py --evaluate` loaded the downloaded model and generated new samples. The original GPU random grid is preserved as `generated_gpu.png`; `generated.png` is the new local inference grid. Summary: `results/metrics/part4_gan.json`.

### Parts 1–3 local validation — 2026-09-10 AEST

- Environment: workspace Python 3.11, PyTorch 2.13.0, torchvision 0.28.0, scikit-learn 1.9.0, NumPy 2.4.6; CPU, CUDA unavailable.
- Commands: `SCIKIT_LEARN_DATA="$PWD/data" OMP_NUM_THREADS=4 .venv/bin/python src/comp3710_lab2/part2.py` and the same command with `part3.py`. Plot verification used the Agg backend and a temporary save/close replacement for interactive `plt.show()`.
- Part 1: full CPU script passed NumPy DFT/FFT and PyTorch square wave, Fourier series and DFT comparisons. The reconstruction and lecture-style spectrum plots were visually checked. GPU timings remain unmeasured.
- LFW data: 1,288 images, 50×37 grayscale pixels in [0, 1], seven classes; lecture split gives 966 training and 322 test images in both Parts 2 and 3.1.
- Part 2: 150-component SVD/PCA, lecture Random Forest parameters; 207/322 correct, **64.29%** test accuracy. Eigenfaces and compactness plots checked. The lecture does not set the Random Forest seed, so later runs can differ; Ariel Sharon had no predicted samples in this run, producing the standard classification-report warning.
- Part 3.1: two 3×3/32-filter convolutions, Adam 0.001, batch size 32, 20 epochs, seed 42; **261/322 correct (81.06%)** on the same held-out split. No test-based parameter tuning was performed.
- Part 3.2: network shape and learning-rate boundary checks passed. Synthetic mini-batches verified backward propagation, parameter updates, checkpoint saving, and fresh-process checkpoint inference including a partial final batch. These checks are not CIFAR-10 accuracy evidence.
- CIFAR-10 download was stopped because the observed rate implied roughly another 40 minutes. No full CIFAR-10 training or Rangpur demonstration has run; >90%, 94% and GPU time targets remain unverified.

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
