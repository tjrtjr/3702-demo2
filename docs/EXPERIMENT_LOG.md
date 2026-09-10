# Experiment log

Actual runs are recorded below. Part 3.2 accuracy, GPU timing and Rangpur demonstration remain unverified; synthetic checks are not evidence of those outcomes. All dates and cluster clock times use AEST.

Result paths below reflect the organization by experiment. Original logs retain their historical paths and job numbers; measured values and model weights were not changed by the move.

## Parts 1–3 local validation

### 2026-08-27 — earlier Part 1 implementation

- Apple Silicon CPU, macOS 26.6.2; Python 3.11.15, NumPy 2.4.6, PyTorch 2.13.0. CUDA was unavailable.
- The then-current Part 1 script ran with `MPLBACKEND=Agg`; input was `float64`, DFT output `complex128`. NumPy naive DFT, PyTorch CPU naive DFT and NumPy FFT agreed (`allclose=True`).
- At N=2048: NumPy FFT ≈0.00001 s, PyTorch CPU naive DFT ≈0.014 s, NumPy naive DFT ≈1.01 s. Fourier reconstruction, spectrum and timing figures were inspected. These are historical measurements; the current GPU comparison is recorded below.

### 2026-09-10 — current Parts 1–3

- Local CPU environment: Python 3.11, PyTorch 2.13.0, torchvision 0.28.0, scikit-learn 1.9.0, NumPy 2.4.6.
- Part 1 completed CPU numerical comparisons for DFT/FFT, square wave and Fourier series. Reconstruction and lecture-style spectrum figures were inspected.
- Parts 2 and 3.1 ran with `SCIKIT_LEARN_DATA="$PWD/data" OMP_NUM_THREADS=4 .venv/bin/python src/comp3710_lab2/part2.py` and the corresponding `part3.py` command. Plot inspection used Agg and a temporary save/close replacement for `plt.show()`.
- LFW: 1,288 images, 50×37 grayscale pixels in [0, 1], seven classes. The lecture split supplies 966 training and 322 test images.
- **Part 2:** 150-component SVD/PCA and lecture Random Forest parameters; **207/322 correct (64.29%)**. Eigenfaces and compactness plots checked. The handout does not set the Random Forest seed, so later runs can differ. This run predicted no Ariel Sharon samples and emitted the standard classification-report warning.
- **Part 3.1:** two 3×3/32-filter convolutions, Adam 0.001, batch size 32, 20 epochs, seed 42; **261/322 correct (81.06%)**. No test-based parameter tuning was performed.
- **Part 3.2:** shape and learning-rate boundary checks passed. Synthetic batches verified backward propagation, parameter updates and fresh-process checkpoint inference, including a partial final batch. The slow CIFAR-10 download was stopped; no full CIFAR-10 training or Rangpur demonstration has run. >90%, 94% and GPU time targets remain unverified.

## OASIS data validation — 2026-09-10

- Used the exact 257.5 MiB archive linked by the handout. ZIP CRC and image/mask filename pairing passed. Images and masks are 256×256; all mask values are 0/85/170/255.
- Preserved course splits: train **9,664 slices / 302 cases**, validation **1,120 / 35**, test **544 / 17**. Case IDs do not overlap.
- Local data root: `data/oasis/keras_png_slices_data/`; Rangpur course root: `/home/groups/comp3710/OASIS/`. The redundant local ZIP was removed after comparing every extracted file's size and CRC; extracted data are retained.

## VAE — 2026-09-10

- Command: `OMP_NUM_THREADS=4 .venv/bin/python src/comp3710_lab2/part4.py`. Completed **30 CPU epochs in 159.79 seconds**, including validation. Validation selected epoch **26**.
- Validation BCE+KL/image: **1075.211**; test BCE+KL/image: **1088.193**. These are objective values, not accuracy percentages.
- Fresh-process `--evaluate` loaded the model and reported **1088.156** test loss/image. Posterior sampling explains the small change. Reconstructions retain brain structure with smoothing; the 2D prior grid shows changing shape and appearance.
- Model: `results/part4_vae/model.pt`. Figures: `results/part4_vae/reconstructions.png`, `results/part4_vae/manifold.png`. Logs and summary: `results/part4_vae/training.log`, `inference.log`, `metrics.json`.

## Rangpur execution — 2026-09-10

- Authorized project directory: `~/comp3710_lab2_20260910/`. The login host resolves to `login0.compute.eait.uq.edu.au`; GPU work ran through `/opt/slurm/bin/` on NVIDIA A100-PCIE-40GB with PyTorch 2.13.0+cu130.
- `a100-test` limits jobs to 20 minutes. Probe **586513** started immediately at 15:53:57, completed in six seconds and passed CUDA tensor arithmetic. Earlier `comp3710` probe **586507** was queued and cancelled before starting. Neither observation guarantees future queue time.
- UNet job **586521** used one A100 and four CPUs. Dependent job **586524** then ran Part 1 timing before GAN training, so neither model trained concurrently with the timing experiment. Both training jobs completed successfully.

### UNet

- Completed **12 epochs**. Validation selected epoch **3** before evaluating the held-out **544** test images.
- Pixel-aggregated test DSC for labels 0/85/170/255: **0.998847 / 0.937156 / 0.947372 / 0.971544**, all >0.9.
- Model: `results/part4_unet/model.pt`; segmentation examples: `results/part4_unet/segmentation.png`, visually checked against ground truth. Full training log: `results/part4_unet/training.log`; summary: `results/part4_unet/metrics.json`.
- Fresh-process local CPU `part4_2.py --evaluate` tested the same 544 images: **0.998847 / 0.937155 / 0.947369 / 0.971544**. Small differences from CUDA AMP are numerical. Log: `results/part4_unet/inference.log`.

### Part 1 GPU timing

- CUDA DFT matched NumPy FFT. At **N=2048**: NumPy FFT **0.000031 s**, PyTorch explicit CUDA DFT **0.000498 s**, NumPy naive DFT **4.108541 s**.
- All four input sizes and correctness checks are recorded in `results/part1/gpu_timing.log`. Times are observed averages under the script's timing protocol, not medians or guarantees for other hardware.

### GAN

- Completed **30 epochs / 4,530 mini-batches** over all **9,664** training slices in **283.34 seconds**. All recorded generator/discriminator losses are finite.
- Model: `results/part4_gan/model.pt`; loss history: `results/part4_gan/history.json`; training log: `results/part4_gan/training.log`; summary: `results/part4_gan/metrics.json`.
- Figures are retained in `results/part4_gan/`. Epoch 10, final GPU samples, loss curve and fresh local CPU samples were inspected: contours, ventricles and tissue patterns visibly vary. This qualitative check does not establish a numerical realism score or exhaustively rule out mode collapse; image realism is judged at the demonstration.
- Fresh-process `part4_3.py --evaluate` loaded the model and generated new samples. `generated_gpu.png` preserves the original GPU grid; `generated.png` is the local inference grid. Inference log: `results/part4_gan/inference.log`.
