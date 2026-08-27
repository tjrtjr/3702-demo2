# Lab 2 coding preparation

Prepared from the complete 13-page Version 2.01 lab sheet and its linked starter notebooks. Part 1 code is in `part1.py`; the CPU path is validated and CUDA timing remains to be run on Rangpur.

## Decisions already made

- Default framework: **PyTorch**. It gives one consistent path across every technical part of the lab.
- Python: **3.11** in a workspace-local `.venv`.
- Reproducibility seed: **42** unless the supplied code requires another seed.
- Models: implement from source code written for this project; do not call a pre-built/pre-trained ResNet, UNet, VAE, or GAN without demonstrator approval.
- Part 4 difficulty: **not selected yet**. The sheet recommends Medium (VAE + UNet), but this remains a student choice because it changes the workload materially.
- Source notebooks: stored unchanged under `references/course_starter/`; copy them before modification.

## Exact acceptance matrix

| Part | Required result | Evidence to retain | Live demonstration requirement |
|---|---|---|---|
| 1 | Fourier square-wave reconstructions; DFT spectrum; framework tensor versions; explicit GPU naive DFT without built-in FFT; timing over changing `N` | plots, correctness check, timing CSV/table, fastest-to-slowest explanation | explain Gibbs phenomenon, complexity, and GPU overhead |
| 2 | LFW PCA with 150 components; eigenface gallery; cumulative variance; RF classifier | split indices, gallery, compactness plot, accuracy and classification report | explain centering, SVD/PCA, projection, and leakage prevention |
| 3.1 | LFW CNN with exactly two 3x3 convolutions, 32 filters each, plus dense classification | architecture summary, train/validation curves, held-out metrics, comparison with RF/PCA | explain tensor layout and every layer |
| 3.2 | Custom CIFAR-10 ResNet-18; >90%; stretch target 94% in about 360 s or faster with mixed precision | code, checkpoint, hardware/time log, test accuracy, one-epoch and inference logs | run inference and one full training epoch on Rangpur |
| 4.1 | Complete the Version Control for Teams using Git short course | completion evidence as permitted by course guidance | follow current Blackboard/Eds instructions |
| 4 Task 1 | Trained OASIS VAE and visualised latent manifold | reconstructions/samples, manifold plot, loss history, checkpoint | explain latent variables and loss |
| 4 Task 2 | OASIS UNet; categorical/one-hot output; DSC >0.9 for every required label | per-class DSC, overlays, split manifest, checkpoint | segment a held-out MRI live |
| 4 Task 3 | OASIS GAN with realistic, varied brains and no unresolved mode collapse | generated grids, loss/training evidence, diversity evidence, checkpoint | justify realism and convergence |

## Part 1 experimental contract

### Required framework functions

Reimplement all three functions named by the PDF with PyTorch operations:

1. `square_wave`: tensor sign/sine expression with no NumPy computation inside the function.
2. `square_wave_fourier`: tensor construction of the requested odd-term Fourier sum; name its argument `n_terms` to avoid confusing it with the number of samples.
3. `naive_dft`: explicit tensor DFT, with a CUDA-capable version that does not call `torch.fft`.

Preserve the NumPy functions as clearly named baselines so plots and numerical comparisons remain easy to audit.

### Implementations to keep distinct

1. `numpy_naive_dft`: literal reference implementation, `O(N^2)`.
2. `numpy_fft`: NumPy's fast implementation, approximately `O(N log N)`.
3. `torch_naive_dft`: explicit tensor DFT, selectable CPU/CUDA, and **never** implemented with `torch.fft`.
4. Optionally keep `torch.fft` only as an additional correctness/performance reference; do not substitute it for item 3.

The PDF's phrase “the three methods” is ambiguous. The safest main table is NumPy naive CPU, NumPy FFT CPU, and tensor naive CUDA; retain tensor naive CPU as an extra row so the tensor conversion itself is demonstrable.

### Correct timing protocol

- Use identical input values and dtype for numerical comparisons where practical.
- Warm up CUDA before measuring.
- Call `torch.cuda.synchronize()` immediately before starting and after ending each CUDA timing region.
- Use `time.perf_counter()`; run repeated trials and report median plus spread, not a single timing.
- State whether host-to-device transfer is excluded (kernel-only) or included (end-to-end). Prefer recording both.
- Suggested sizes: `N = 256, 512, 1024, 2048`, stopping early if the literal CPU implementation is impractical.
- Check each explicit DFT against the FFT with documented tolerances.
- A vectorised DFT matrix still costs `O(N^2)` memory and work. Estimate memory before increasing `N` on the GPU.

### Interpretation to be ready to explain

- Higher odd harmonics sharpen transitions and reduce error away from discontinuities.
- Gibbs overshoot near each jump narrows but does not disappear simply by adding more terms.
- With 50 **terms**, the supplied function includes odd harmonics 1 through 99; the PDF plot only displays 0-50 Hz.
- An FFT wins asymptotically because it reuses structure instead of evaluating every sample/frequency pair.
- A GPU naive DFT may lose for small arrays because launch, synchronisation, and transfer costs dominate, but should scale better than Python loops once enough parallel work exists.

## Part 2 data and evaluation contract

- Load `fetch_lfw_people(min_faces_per_person=70, resize=0.4)` into a project-local ignored cache when possible.
- Preserve a deterministic 25% held-out test split with `random_state=42`; add `stratify=y` because the PDF comment claims a stratified split even though its code omits it.
- Save the split indices so Part 2 and Part 3.1 compare on the same test subjects/images.
- Compute the mean and SVD from training data only. Apply the training mean/components to validation/test data.
- Use 150 components and reshape the first 12 for the required 3x4 eigenface gallery.
- Use `X_train.shape[0] - 1` for an absolute explained-variance denominator. The PDF uses total `n_samples`; this common factor cancels in its ratio, but is not the rigorous definition.
- Build the supplied RF baseline with 150 trees, depth 15, and 150 max features; add `random_state=42` for reproducibility.

## Part 3 implementation boundary

### LFW CNN

The two `3x3`, 32-filter convolutions are hard constraints. Pooling, activations, dropout, and dense width remain design choices. Use channel-first input `[batch, 1, height, width]` in PyTorch. LFW values are already in `[0, 1]`; verify this rather than blindly renormalising. Keep a validation subset separate from the final test set.

### CIFAR-10 / DAWNBench

Use a CIFAR stem and implement the residual blocks and `[2, 2, 2, 2]` ResNet-18 stage layout in this project. A useful official reference from the sheet uses 35 epochs, batch size 128, SGD with momentum 0.9, weight decay `5e-4`, a one-cycle schedule, CIFAR mean/std normalisation, random horizontal flip, reflect padding by four pixels, and random 32x32 crop. Treat these as a baseline to test, not hidden assessment requirements.

For the stretch mark, use CUDA automatic mixed precision, record actual A100 wall-clock time and the exact timing boundary, and save a best checkpoint plus a fast demo checkpoint. Prepare two stable commands before the demo: one for a single epoch and one for inference.

## Part 4 gates and unresolved questions

Before implementing OASIS models, inspect the cluster files and resolve:

1. Actual filenames, tensor/image shapes, label values, subject grouping, and any official split.
2. Whether the expected model is 2D or 3D.
3. Whether “all labels” for DSC includes background and whether DSC is per-volume, macro averaged, or globally aggregated.
4. Whether one-hot is required for both stored targets and model output/loss presentation.
5. What the demonstrator counts as a forbidden pre-built model.
6. Whether the 360-second DAWNBench comparison includes loading, compilation, evaluation, and checkpoint I/O.

Avoid slice-level leakage: if OASIS contains multiple slices/volumes per subject, split by subject before deriving training samples.

## Known external blockers

- The PDF does not contain the due date; obtain it from the current course instance.
- GitHub CLI is installed but its saved token is invalid. Re-authentication is required before creating/pushing the student's own remote repository.
- Rangpur access, SLURM modules, `/home/groups/comp3710/` permissions, and the OASIS schema have not yet been verified.
- Local execution currently has no CUDA device, so GPU timings cannot be accepted from this machine.

## Recommended coding order

1. Part 1 reference plots, correctness tests, and benchmark harness.
2. Part 2 shared LFW data split, PCA/RF baseline, plots, and metrics.
3. Part 3.1 CNN using exactly the same held-out LFW test set.
4. Rangpur environment/data reconnaissance before committing to cluster dependencies.
5. Part 3.2 custom ResNet-18, then its one-epoch/inference demo paths.
6. Part 4 only after selecting Easy/Medium/Hard and inspecting OASIS.

This order produces demonstrable, independently committable milestones and prevents the expensive cluster tasks from blocking the early marks.
