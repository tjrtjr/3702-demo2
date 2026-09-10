# Experiment log

Actual runs are recorded below. Part 3.2 training, inference and a single epoch have now been verified on Rangpur; the student must still perform the required live demonstration in front of the tutor. All dates and cluster clock times use AEST.

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
- **Part 3.2 preliminary checks:** shape and learning-rate boundary checks passed. Synthetic batches verified backward propagation, parameter updates and fresh-process checkpoint inference, including a partial final batch. The first slow CIFAR-10 download was stopped. These preliminary checks did not establish accuracy; the subsequent real-data experiment is recorded below.

## Part 3.2 CIFAR-10 on Rangpur — 2026-09-10

- Code: commit `e726316888821daa2fa4fc4a14f67f4c653e0bfe`, `src/comp3710_lab2/part3_2.py` SHA256 `5d9b627d6a498c9076c8c38b63a2d082a6039a7ef16c5f2fa34a4cf92a58d049`. No training-code or hyperparameter changes were made for this run.
- Data: original CIFAR-10 archive from the UCSD mirror, MD5 `c58f30108f718f92721af3b95e74349a`; torchvision validated all batches. Original **50,000 training / 10,000 test** split retained. Verified data are available on Rangpur and locally.
- Hardware/environment: NVIDIA A100-PCIE-40GB on `a100-b`, PyTorch **2.13.0+cu130**, torchvision **0.28.0+cu130**, four allocated CPUs, `OMP_NUM_THREADS=4`, CUDA automatic mixed precision. Course reference ResNet-18, batch size **128**, **35 epochs**, original SGD/augmentation/learning-rate schedule.
- Single-epoch job **586613**: submitted 17:49:03, started 17:58:05, completed 17:58:31; exit `0:0`. `--epochs 1 --checkpoint results/part3_2/demo/model.pt` trained all 50,000 images in **7.60 seconds**, cross entropy **1.66890**, test accuracy **43.14%**. Fresh-process inference reproduced 43.14%. This check established learning, complete data traversal and checkpoint reload; it was not the final accuracy benchmark.
- Full job **586620**: submitted and started 18:06:03, completed 18:10:12; exit `0:0`. Ran `python -u src/comp3710_lab2/part3_2.py`, completed all **35 epochs**, final training cross entropy **0.02891**. Test accuracy **9,419/10,000 = 94.19%**; training time **228.94792494 seconds (3.82 minutes)**. The final 35-epoch checkpoint was evaluated; no epoch or hyperparameter was selected by test accuracy.
- Timing includes the training loop's data loading, augmentation and CUDA synchronization, and excludes data download, model setup, test evaluation, checkpoint saving and queue wait. The complete Slurm job took **4 minutes 9 seconds**. The 94% accuracy and below-360-second numerical targets were met on the reported A100; this is not a V100 measurement.
- A separate process in the full GPU job ran `python -u src/comp3710_lab2/part3_2.py --evaluate`: all 10,000 images again gave **94.19%**, with **1.18 seconds** measured inference time (post-training inference was **0.87 seconds**).
- Artifacts: `results/part3_2/model.pt`, `training.log`, `inference.log`, `metrics.json`; the single-epoch model and logs are under `results/part3_2/demo/`. Checkpoint SHA256 values and Slurm timestamps are recorded in `metrics.json`. The student must still run inference and one epoch in front of the tutor.

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
