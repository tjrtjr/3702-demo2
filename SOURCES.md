# Sources and AI assistance

## Course code and required implementations

The authoritative handout is Shekhar “Shakes” Chandra, *Lab Demonstration 2: Pattern Recognition*, Version 2.01, 2026, distributed through the course. Course materials are not republished in this repository.

- **Part 1:** NumPy functions and initial plotting structure follow the course-provided [square-wave example](https://colab.research.google.com/drive/1dO6YBhLAuGT69ZVqpkmxEL1raEAEWIft?usp=sharing) and [naive DFT/FFT example](https://colab.research.google.com/drive/1kG44bMAVQb_BWgn2DtcO81MjfYeTaJc_?usp=sharing). The PyTorch tensor versions, CUDA execution and size benchmark implement the requested extension.
- **Part 2:** `part2.py` follows handout pages 6–9, including the original data split and Random Forest parameters. Transcription errors were corrected.
- **Part 3.1:** `part3.py` follows the LFW preprocessing and two 3×3/32-filter convolution constraints on page 9. The handout does not provide the complete network: ReLU, 2×2 pooling, a 128-unit hidden dense layer, batch size 32, 20 epochs and Adam learning rate 0.001 fill in necessary details and are identified in the code.
- **Part 3.2:** `part3_2.py` ports the Appendix B reference to PyTorch: [network](https://github.com/shakes76/jax-vision/blob/main/networks.py), [preprocessing](https://github.com/shakes76/jax-vision/blob/main/preprocess.py), and [training](https://github.com/shakes76/jax-vision/blob/main/test_resnet_cifar.py) (Apache-2.0). It retains the reference's 64/128/256/256 channel variant, projection shortcuts, 35 epochs, batch size 128, SGD and linear one-cycle schedule. CUDA mixed precision implements the additional lab requirement. The PyTorch loader includes the last partial batch instead of batching a repeated TF dataset. The unchanged port was trained on Rangpur A100 for 35 epochs: 94.19% test accuracy in 228.95 seconds; full results are in the experiment log.

## Part 4 sources and adaptations

The handout specifies tasks but provides no complete VAE, UNet or GAN source code. These are implementations of the requested tasks, not lecture-original code:

| File | Source and necessary adaptation |
|---|---|
| `oasis.py` | Reads the course PNG archive's extracted folders and preserves its train/validate/test split. Image/mask pairs, mask values 0/85/170/255 and disjoint subject groups were checked. |
| `part4.py` | [Official PyTorch VAE example](https://github.com/pytorch/examples/blob/main/vae/main.py): retains 400-unit hidden layers, reparameterization, BCE + KL and Adam 0.001. Uses 64×64 grayscale OASIS inputs and two latent variables for direct manifold plotting. |
| `part4_2.py` | [U-Net architecture linked by the handout](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/): same padding preserves 256×256 output size; initial width is 32. Uses four-channel one-hot targets, categorical cross entropy and Adam 0.001. Validation selects the checkpoint; test DSC is aggregated over pixels for each of four labels, including background. |
| `part4_3.py` | [Official PyTorch DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html): retains the 64×64 architecture, initialization, BCE and Adam settings. OASIS images use one grayscale channel, resize to 64×64 and scale to [-1, 1]. |

Measured results and limitations are recorded in [the experiment log](docs/EXPERIMENT_LOG.md). Training outcomes do not replace the Git short course or the student's demonstration and explanation.

## Data and other references named by the handout

- [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
- [scikit-learn Eigenfaces example](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)
- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
- [DAWNBench CIFAR-10 benchmark](https://dawn.cs.stanford.edu/benchmark/index.html#cifar10-train-time)
- [Shakes' JAX vision reference](https://github.com/shakes76/jax-vision)
- [UQ Rangpur compute information](https://student.eait.uq.edu.au/infrastructure/compute/)
- [OASIS course download](https://filesender.aarnet.edu.au/?s=download&token=e49c9831-d72f-422f-9b2c-a06283271b9a)

For the CIFAR-10 run, the original archive was obtained from the [UCSD university mirror](https://cseweb.ucsd.edu/~weijian/static/datasets/cifar/) because the original host downloaded slowly. Its MD5 `c58f30108f718f92721af3b95e74349a` matches torchvision's official archive checksum; all training/test batches passed torchvision's integrity checks. The original 50,000/10,000 split is unchanged.

## AI assistance record

The following work used OpenAI Codex. Checks listed here were performed by Codex; they are not claims of independent student verification or authorship.

| Date | Task and material affected | Verification |
|---|---|---|
| 2026-08-27 | Read the 13-page handout and linked examples; prepare the initial project; implement Part 1 and Chinese code comments. | PDF text and pages checked, course examples compared, CPU numerical comparisons and plots checked. |
| 2026-09-10 | Align Parts 1–3 with the handout, fix Part 2 transcription errors, implement required CNNs and document commands. | Parts 1, 2 and 3.1 ran locally. Part 3.2 received structural, backward-pass and checkpoint checks using synthetic inputs only. |
| 2026-09-10 | Implement Part 4 using the references above; validate OASIS; run VAE locally and UNet/GAN on Rangpur; save models, logs and figures. | Course splits, labels and pairing checked; three models trained and reloaded in fresh processes; result images inspected. Part 1 GPU timing also completed. |
| 2026-09-10 | Simplify documentation, remove unused scaffolding and course example copies, group results by experiment, and document the handout's demonstration requirements. | Result files and model weights preserved; script output paths updated without algorithm changes. Three migrated models passed fresh-process inference, including all 544 UNet test images. |
| 2026-09-10 | Execute the existing Part 3.2 code on Rangpur using original CIFAR-10 data and course-reference parameters; save results and update demonstration instructions. | Archive and batches verified; real-data single-epoch training/reload passed; 35 A100 epochs achieved 94.19% in 228.95 seconds; independent GPU inference reproduced 94.19%. |
| 2026-09-15 | Audit the existing code and results against the handout and clarify the demonstration order, required evidence, implementation choices and auxiliary checks in `results/README.md`. | Handout text and relevant rendered pages checked; scripts, metrics and logs compared; Appendix B reference rechecked. Documentation only; no new training or result deletion. |

The student must review and be able to explain the retained code and results; AI-generated contributions should not be represented as independently authored.
