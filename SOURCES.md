# Sources and AI assistance

Keep this file current as code is written. Clearly distinguish course-provided code, adapted external ideas, and original implementation.

## Course source

- Shekhar “Shakes” Chandra, *Lab Demonstration 2: Pattern Recognition*, Version 2.01, 2026. Local course PDF: `COMP3710_Lab_2_2026_v2.01.pdf`.
- Course-provided square-wave Colab: <https://colab.research.google.com/drive/1dO6YBhLAuGT69ZVqpkmxEL1raEAEWIft?usp=sharing>
- Course-provided Gemini naive DFT/FFT Colab: <https://colab.research.google.com/drive/1kG44bMAVQb_BWgn2DtcO81MjfYeTaJc_?usp=sharing>

Unmodified local copies of the two notebooks are kept under `references/course_starter/` and ignored by Git. Any modified assessment notebook must identify which cells were adapted.

The NumPy functions and initial plotting structure in `part1.py` are adapted from these course sources. The PyTorch tensor versions, CUDA execution, and size benchmark are the submitted extension.

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
| | | | | |

For future entries, summarise the prompt/task and the substantive suggestions used. Do not claim AI-generated code as independently authored; review, test, and be able to explain every retained line.
