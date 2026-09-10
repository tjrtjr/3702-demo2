# COMP3710 Lab Demonstration 2

本项目包含 Part 1–4 的代码。讲义有代码的部分沿用讲义；Part 1 的 PyTorch 改写和 Part 3.1 的网络是讲义要求自行实现的内容。Part 3.1 未指定的网络细节和训练参数已在代码中注明。Part 3.2 根据附录 B 链接的 [jax-vision 课程示例](https://github.com/shakes76/jax-vision)移植为 PyTorch。Part 4 的讲义没有完整代码，所用基础实现和必要适配在各脚本中注明，不能称为讲义原代码。

## 在本机运行

打开终端，进入项目并启用已有环境，无需重新创建 `.venv`：

```bash
cd "/Users/junrongtang/Desktop/3710 demo2"
source .venv/bin/activate
export SCIKIT_LEARN_DATA="$PWD/data"
```

然后按需逐条运行：

```bash
# Part 1：方波、傅里叶级数、DFT/FFT 和张量计时
python src/comp3710_lab2/part1.py

# Part 2：Eigenfaces/PCA 和随机森林
python src/comp3710_lab2/part2.py

# Part 3.1：LFW 两层 CNN（训练 20 个 epoch 后输出测试准确率）
python src/comp3710_lab2/part3.py
```

Part 1、Part 2 会弹出图窗；看完关闭图窗，程序才会继续或退出。Part 2 需要依次关闭两次图窗才会运行随机森林。数据不存在时，Part 2、Part 3 会下载 LFW，Part 3.2 会下载 CIFAR-10，均存放在项目的 `data/` 下。

Part 3.2 的完整训练与推理：

```bash
# 完整训练 35 个 epoch，保存 results/part3_2.pt
python src/comp3710_lab2/part3_2.py

# 加载上述训练结果进行推理
python src/comp3710_lab2/part3_2.py --evaluate --checkpoint results/part3_2.pt
```

只运行一个 epoch 并检查其结果：

```bash
python src/comp3710_lab2/part3_2.py --epochs 1 --checkpoint results/part3_2_demo.pt
python src/comp3710_lab2/part3_2.py --evaluate --checkpoint results/part3_2_demo.pt
```

推理命令需要对应的模型文件；一个 epoch 的运行用于检查流程，不能证明达到讲义准确率要求。Part 3.2 在本机 CPU 上训练较慢。

## 讲义中的 GPU 要求

- **Part 1（第 5 页）**：要求实际比较 GPU 张量 DFT、NumPy 朴素 DFT 和 NumPy FFT 的耗时，并改变数据大小。讲义此处未限定必须在 Rangpur；当前脚本使用 CUDA，没有 CUDA 时会留下待测项。
- **Part 2、Part 3.1**：没有明确要求 GPU，可在本机 CPU 上运行。
- **Part 3.2（第 10 页）**：超过 90% 准确率且训练较快（通常集群上少于 30 分钟）占 1 分；演示时在 Rangpur 运行推理和一个训练 epoch 占 1 分；混合精度达到 94%，且时间与 V100 上约 360 秒相当或更快，占 2 分。因此，完整完成本部分需要 Rangpur 实测。当前尚未验证准确率和 GPU 耗时目标。

Part 1 的本次 A100 实测已完成，日志为 `results/metrics/part1_gpu.log`。N=2048 时：NumPy FFT 0.000031 秒、GPU 张量 DFT 0.000498 秒、NumPy 朴素 DFT 4.108541 秒，三者数值一致。

## Part 4 当前范围

Part 4 三个模型的代码和本次训练已完成，**完整评分要求尚未全部完成**。VAE 已在本机完成 30 轮训练和重新加载推理；UNet 已在 Rangpur A100 训练 12 轮，以验证集选择第 3 轮模型，测试集四类 DSC 为 **0.998847 / 0.937156 / 0.947372 / 0.971544**，全部 >0.9。GAN 已完成 30 轮 A100 训练（约 283 秒），生成图和损失曲线已保存并检查；生成图的最终真实度由演示教师评判。当前讲义第 10–12 页只有任务要求，没有完整模型代码。

- **4.1（1 分）**：完成 edX 的 *Version Control for Teams using Git* 课程，入口以 Blackboard/Eds 公告为准。
- **4.4（最多 7 分，逐级累计）**：VAE 训练和流形可视化最多 3 分；再完成 UNet 分割最多 5 分；再完成 GAN 脑图像生成最多 7 分。UNet 要求各标签 DSC > 0.9、类别（one-hot）输出、结果可视化及现场测试集推理；GAN 要有逼真的 OASIS 生成结果和训练证据。

讲义还要求在本人 GitHub 账号下建立项目、保留相关提交记录、注明来源，并能解释代码与结果。此前筹备文档中的额外建议（例如保存划分索引、逐次记录环境）不是讲义逐项规定的硬性要求。

Git 短课完成情况尚未提供；当前项目未配置 GitHub remote，也尚未上传至 GitHub。模型训练结果不能替代这两项要求。

## Part 4 数据与结果

已从讲义原链接下载并检查数据。本机路径为 `data/oasis/keras_png_slices_data/`，Rangpur 上直接使用 `/home/groups/comp3710/OASIS/`，无需重复下载：

| 划分 | 图像文件夹 | 标签文件夹 | 切片数 |
|---|---|---|---:|
| 训练 | `keras_png_slices_train` | `keras_png_slices_seg_train` | 9,664 |
| 验证 | `keras_png_slices_validate` | `keras_png_slices_seg_validate` | 1,120 |
| 测试 | `keras_png_slices_test` | `keras_png_slices_seg_test` | 544 |

所有图像为 256×256 灰度 PNG。标签值为 0、85、170、255，代码转为四类 one-hot；图像和标签一一对应，三组病例没有重叠。沿用课程提供的划分，不重新随机拆分切片。

本机命令（UNet 和 GAN 完整训练宜在 GPU 上运行）：

```bash
python src/comp3710_lab2/part4.py
python src/comp3710_lab2/part4_2.py
python src/comp3710_lab2/part4_3.py
```

加载已保存模型时，在相应命令后加 `--evaluate`。可以用 `--epochs 1 --checkpoint results/part4_2_canary.pt` 做单轮检查，避免覆盖正式模型。

| 实验 | 默认模型 | 结果 |
|---|---|---|
| VAE | `results/part4.pt` | `results/part4_reconstructions.png`、`results/part4_manifold.png` |
| UNet | `results/part4_2.pt` | `results/part4_2.png`；终端逐类 DSC 和是否全部 >0.9 |
| GAN | `results/part4_3.pt` | `results/figures/part4_3/` 下的生成图和损失曲线；`results/part4_3.json` 为损失历史 |

## 在 Rangpur 上运行 Part 4

已核实配置的登录地址 `rangpur.compute.eait.uq.edu.au`、SLURM 路径 `/opt/slurm/bin/`，并实际在 `a100-test` 分区运行过 A100 张量计算。`comp3710` 分区可运行较长实验；`a100-test` 每个作业最长 20 分钟，适合检查或短实验，排队时间随资源变化。

代码已上传到 `~/comp3710_lab2_20260910/`，可在登录节点执行以下命令提交训练。用户已确认此账号和实验目录；登录节点本身不运行 GPU 训练。

```bash
ssh rangpur.compute.eait.uq.edu.au
cd ~/comp3710_lab2_20260910
mkdir -p results runs

# UNet 正式训练；日志文件包含 SLURM 作业号
/opt/slurm/bin/sbatch --account=comp3710 --partition=comp3710 \
  --gres=gpu:a100:1 --cpus-per-task=4 --time=03:00:00 \
  --job-name=lab2-unet --output=runs/unet-%j.log \
  --wrap='/home/Student/s4983489/miniconda3/envs/torch/bin/python -u src/comp3710_lab2/part4_2.py --data-dir /home/groups/comp3710/OASIS --checkpoint results/part4_2.pt'
```

VAE 或 GAN 训练只需将上述脚本、模型文件、作业名和日志名分别改为 `part4.py` / `part4.pt` / `vae` 或 `part4_3.py` / `part4_3.pt` / `gan`。先检查流程时，可用 `--partition=a100-test --time=00:20:00`，并在 Python 命令末尾加 `--epochs 1`，使用单独的 canary 模型路径。

```bash
# 查看自己的作业和训练日志
/opt/slurm/bin/squeue -u "$USER"
tail -f runs/unet-实际作业号.log

# 现场加载 UNet，在课程测试集推理并输出 DSC 和分割图
/opt/slurm/bin/srun --account=comp3710 --partition=a100-test \
  --gres=gpu:a100:1 --cpus-per-task=4 --time=00:20:00 \
  /home/Student/s4983489/miniconda3/envs/torch/bin/python \
  src/comp3710_lab2/part4_2.py --data-dir /home/groups/comp3710/OASIS \
  --evaluate --checkpoint results/part4_2.pt
```

模型和图像保留在集群项目的 `results/`，日志保留在 `runs/`。在本机项目终端下载到独立目录，保留已有本机结果：

```bash
mkdir -p results/rangpur
scp -r rangpur.compute.eait.uq.edu.au:~/comp3710_lab2_20260910/results results/rangpur/
scp -r rangpur.compute.eait.uq.edu.au:~/comp3710_lab2_20260910/runs results/rangpur/
```

本次 UNet 和 GAN 的模型、图片和训练日志已下载到本机相应的 `results/` 路径；VAE 在本机训练，结果也在 `results/`。三者均已完成本机独立进程的 `--evaluate` 检查，展示已有结果无需重新训练。
