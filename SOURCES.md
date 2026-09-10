# Sources and AI assistance

Keep this file current as code is written. Clearly distinguish course-provided code, adapted external ideas, and original implementation.

## Course source

- Shekhar “Shakes” Chandra, *Lab Demonstration 2: Pattern Recognition*, Version 2.01, 2026. Local course PDF: `COMP3710_Lab_2_2026_v2.01.pdf`.
- Course-provided square-wave Colab: <https://colab.research.google.com/drive/1dO6YBhLAuGT69ZVqpkmxEL1raEAEWIft?usp=sharing>
- Course-provided Gemini naive DFT/FFT Colab: <https://colab.research.google.com/drive/1kG44bMAVQb_BWgn2DtcO81MjfYeTaJc_?usp=sharing>

Unmodified local copies of the two notebooks are kept under `references/course_starter/` and ignored by Git. Any modified assessment notebook must identify which cells were adapted.

The NumPy functions and initial plotting structure in `src/comp3710_lab2/part1.py` are adapted from these course sources. The PyTorch tensor versions, CUDA execution, and size benchmark implement the requested extension.

`src/comp3710_lab2/part2.py` follows pages 6–9, including the original split and Random Forest parameters. `src/comp3710_lab2/part3.py` follows the LFW preprocessing and CNN constraints on page 9. The sheet does not supply the complete CNN: ReLU, 2×2 pooling, a 128-unit hidden dense layer, batch size 32, 20 epochs, and Adam's default learning rate 0.001 fill in those necessary details.

`src/comp3710_lab2/part3_2.py` ports the Appendix B course reference to PyTorch: [network](https://github.com/shakes76/jax-vision/blob/main/networks.py), [preprocessing](https://github.com/shakes76/jax-vision/blob/main/preprocess.py), and [training](https://github.com/shakes76/jax-vision/blob/main/test_resnet_cifar.py). It retains that reference's 64/128/256/256 channel variant, projection shortcuts, 35 epochs, batch size 128, SGD and linear one-cycle schedule. CUDA mixed precision implements the additional lab requirement. The PyTorch loader includes the last partial batch instead of batching a repeated TF dataset. This implementation has not yet demonstrated the required CIFAR-10 accuracy or Rangpur GPU times.

## Part 4 implementation sources

The lab sheet contains task requirements, not complete VAE, UNet or GAN source code. The following are necessary implementations of those tasks, not additional lab tasks or claimed lecture-original code:

- `oasis.py`: reads the course-provided PNG archive and preserves its train/validate/test directories. Image/mask pairing and all mask values (0, 85, 170, 255) were checked; the original subject groups do not overlap.
- `part4.py`: adapts the [official PyTorch VAE example](https://github.com/pytorch/examples/blob/main/vae/main.py), retaining its 400-unit hidden layers, reparameterization, BCE + KL objective and Adam 0.001. OASIS-specific choices are 64×64 grayscale inputs and two latent variables for direct manifold plotting, with the original course data splits.
- `part4_2.py`: implements the [U-Net architecture linked by the lab](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). Same padding keeps the 256×256 input/output size; the initial channel count is 32. It uses four-channel one-hot targets, categorical cross entropy and Adam 0.001. Validation chooses the checkpoint; test evaluation reports the pixel-aggregated DSC for each of the four labels, including background.
- `part4_3.py`: adapts the [official PyTorch DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html), retaining its 64×64 architecture, initialization, BCE loss and Adam settings. Input/output channels are changed to grayscale OASIS, and the original 256×256 images are resized to 64×64 and mapped to [-1, 1].

Only trained outputs and measured metrics can support the Part 4 assessment; architecture or single-batch checks do not establish segmentation accuracy, realistic generation or absence of mode collapse. The Git short course and the student's own GitHub project remain separate assessment requirements.

## Data and reference implementations named by the lab sheet

- Labeled Faces in the Wild: <http://vis-www.cs.umass.edu/lfw/>
- scikit-learn Eigenfaces example: <https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html>
- CIFAR-10: <http://www.cs.toronto.edu/~kriz/cifar.html>
- DAWNBench CIFAR-10 benchmark: <https://dawn.cs.stanford.edu/benchmark/index.html#cifar10-train-time>
- Shakes' JAX vision reference (Apache-2.0): <https://github.com/shakes76/jax-vision>
- U-Net project/paper page: <https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>
- UQ Rangpur compute information: <https://student.eait.uq.edu.au/infrastructure/compute/>
- OASIS course download link from the lab sheet: <https://filesender.aarnet.edu.au/?s=download&token=e49c9831-d72f-422f-9b2c-a06283271b9a>

## AI assistance log

| Date | Tool/model | Purpose | Material affected | Human verification performed |
|---|---|---|---|---|
| 2026-08-27 | OpenAI Codex | Read the full PDF, extract requirements, audit ambiguities, and prepare the coding scaffold | `README.md`, `docs/`, `requirements.txt`, `.gitignore`, `SOURCES.md` | All 13 pages were text-extracted and visually checked; official linked notebooks were downloaded and compared with the PDF |
| 2026-08-27 | OpenAI Codex | Implement Part 1 from the course code, add the required PyTorch/CUDA DFT and timing comparison, and write Chinese explanatory comments | `part1.py`, `README.md`, `docs/LAB2_PREPARATION.md`, `SOURCES.md` | Full CPU run completed; three numerical comparisons returned `True`; all three figures were rendered and inspected; CUDA branch remains to be run on Rangpur |
| 2026-09-10 | OpenAI Codex | Follow the lab sheet for Parts 1–3 without extra tasks: align the Part 1 spectrum plot, fix Part 2 transcription errors, implement the required CNNs | `src/comp3710_lab2/part1.py`, `part2.py`, `part3.py`, `part3_2.py`; run instructions and this log | Codex checked the relevant PDF pages and linked course source; Parts 1, 2 and 3.1 ran locally. Part 3.2 received structural, forward/backward and checkpoint checks using synthetic inputs only. Student verification and Rangpur runs remain outstanding. |
| 2026-09-10 | OpenAI Codex | Implement the three requested Part 4 tasks from the lab requirements and identified basic references; inspect course OASIS data and prepare Rangpur experiments | `oasis.py`, `part4.py`, `part4_2.py`, `part4_3.py`, run instructions and result files | All course mask values/pairs and subject splits checked. VAE trained for 30 CPU epochs and checkpoint inference verified; its reconstructions and manifold inspected. UNet/GAN passed real-image single-batch and checkpoint checks. Full GPU experiment outcomes are recorded separately in the experiment log. Student review and explanation remain required. |

For future entries, summarise the prompt/task and the substantive suggestions used. Do not claim AI-generated code as independently authored; review, test, and be able to explain every retained line.
