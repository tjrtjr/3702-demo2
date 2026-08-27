# COMP3710 Lab Demonstration 2

This repository is prepared for the 2026 Pattern Recognition Lab Demonstration 2. Part 1 is implemented as a single lecture-aligned Python script; the remaining requirements, environment, references, and experiment protocol are also recorded here.

PyTorch is the default framework because it can cover the tensor DFT, LFW CNN, CIFAR-10 ResNet-18, mixed-precision A100 training, and the optional OASIS tasks without switching frameworks. This is a working default, not a restriction imposed by the lab sheet.

## Assessment map

| Part | Work | Marks |
|---|---|---:|
| 1 | Fourier reconstruction, DFT/FFT, tensor/GPU timing | 1 |
| 2 | LFW Eigenfaces/PCA and Random Forest | 1 |
| 3.1 | Two-layer LFW CNN | 1 |
| 3.2 | CIFAR-10 custom ResNet-18 / DAWNBench | 4 |
| 4.1 | Advanced Git course | 1 |
| 4.4 | Cumulative OASIS track: VAE (up to 3/7), VAE + UNet (up to 5/7), or all three including GAN (up to 7/7) | 7 |

The exact implementation contract, ambiguities, and acceptance checks are in [docs/LAB2_PREPARATION.md](docs/LAB2_PREPARATION.md). Record every meaningful run in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md), and keep citations/AI assistance current in [SOURCES.md](SOURCES.md).

## Local setup

The tested local route uses Python 3.11 and a workspace-local virtual environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python -c "import numpy, sklearn, torch, torchvision; print(torch.__version__); print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
python part1.py
```

Local CPU execution is enough for Parts 1-3 smoke tests. The required CUDA timing, DAWNBench training, and OASIS work must be validated on UQ's Rangpur cluster; do not treat a local Apple GPU result as the required cluster demonstration.

## Project layout

```text
part1.py                 Part 1 Fourier, DFT, PyTorch, and timing work
notebooks/               optional exploratory work
src/comp3710_lab2/       reusable models, data, metrics, and training code
scripts/                 command-line train/evaluate/infer entry points
slurm/                   Rangpur job scripts and demo commands
tests/                   fast correctness and shape tests
results/figures/         selected figures suitable for Git
results/metrics/         compact machine-readable metrics suitable for Git
references/course_starter/  unmodified local copies of the two linked Colabs
docs/                    requirements and experiment evidence
```

Large datasets, checkpoints, transient logs, the course PDF, and the copied starter notebooks are deliberately ignored by Git. Curated figures and metrics are not ignored because they are part of the demonstration evidence.

## Non-negotiable workflow

1. Make a small, meaningful commit at each completed milestone; Part 4 receives no marks without a project in the student's own GitHub account and relevant commit history.
2. Implement ResNet-18 and any Part 4 network explicitly unless the demonstrator approves a pre-built model.
3. Fit preprocessing/PCA only on training data, preserve split indices, fix random seeds, and log the exact environment and hardware.
4. Save runnable inference commands and one-epoch cluster commands before the demonstration.
5. Cite course, web, and AI-derived material and be prepared to explain every layer, result, and design decision.

## First coding milestone

Run `python part1.py`. The script keeps the lecture's NumPy functions and plots, then reimplements `square_wave`, `square_wave_fourier`, and `naive_dft` with PyTorch operations. It verifies the explicit tensor DFT without `torch.fft` and compares timings across several input sizes. CUDA timing is filled automatically when the same script runs on Rangpur.
