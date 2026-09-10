# COMP3710 Lab Demonstration 2

Part 1–4 的 Python 脚本。讲义提供代码的部分沿用讲义；要求自行实现的部分和必要适配均在代码及 [SOURCES.md](SOURCES.md) 中注明。Part 4 讲义没有完整模型代码，使用所注明的基础参考实现。

**给 tutor 展示时，从 [结果与展示入口](results/README.md) 开始：讲义要求、对应图像、展示顺序和现场命令都在那里。**

| 路径 | 内容 |
|---|---|
| `src/comp3710_lab2/` | 各部分脚本和 OASIS 数据读取代码 |
| `results/` | 实测日志、指标、结果图；模型权重不纳入 Git |
| `docs/EXPERIMENT_LOG.md` | 已完成实验及尚未验证的要求 |
| `SOURCES.md` | 讲义、代码与数据来源，AI 使用记录 |
| `requirements.txt` | Python 运行依赖 |

数据、模型权重、讲义 PDF 和本机虚拟环境不上传 GitHub。课程原始资料入口见 [SOURCES.md](SOURCES.md)。

## 本机运行

以下命令均在项目根目录执行。已有 `.venv` 时直接启用：

```bash
source .venv/bin/activate
export SCIKIT_LEARN_DATA="$PWD/data"
```

首次配置环境时，先执行 `python3.11 -m venv .venv`，再启用环境并运行 `python -m pip install -r requirements.txt`。

```bash
python src/comp3710_lab2/part1.py    # 傅里叶级数、DFT/FFT、CPU/GPU 计时
python src/comp3710_lab2/part2.py    # Eigenfaces/PCA、随机森林
python src/comp3710_lab2/part3.py    # Part 3.1：LFW CNN，训练 20 轮

# Part 3.2：CIFAR-10，完整训练 35 轮并保存模型
python src/comp3710_lab2/part3_2.py
python src/comp3710_lab2/part3_2.py --evaluate --checkpoint results/part3_2/model.pt
```

Part 1、Part 2 会弹出图窗，关闭后程序继续；Part 2 需依次关闭两个图窗后才运行随机森林。LFW 和 CIFAR-10 缺失时自动下载到 `data/`。

Part 3.2 单轮检查使用独立模型文件：

```bash
python src/comp3710_lab2/part3_2.py --epochs 1 --checkpoint results/part3_2/demo/model.pt
python src/comp3710_lab2/part3_2.py --evaluate --checkpoint results/part3_2/demo/model.pt
```

单轮检查不能证明达到准确率要求。**Part 3.2 尚未完成真实 CIFAR-10 训练和 Rangpur 演示**，>90%、94% 及 GPU 耗时目标仍未验证。

## Part 4 数据、运行和保存

本机 OASIS 路径为 `data/oasis/keras_png_slices_data/`；Rangpur 已有课程数据 `/home/groups/comp3710/OASIS/`，无需重复下载。首次在其他机器运行时，从 [课程数据链接](https://filesender.aarnet.edu.au/?s=download&token=e49c9831-d72f-422f-9b2c-a06283271b9a)取得压缩包并解压到本机路径，或用 `--data-dir` 指定位置。

| 划分 | 图像文件夹 | 标签文件夹 | 切片数 |
|---|---|---|---:|
| 训练 | `keras_png_slices_train` | `keras_png_slices_seg_train` | 9,664 |
| 验证 | `keras_png_slices_validate` | `keras_png_slices_seg_validate` | 1,120 |
| 测试 | `keras_png_slices_test` | `keras_png_slices_seg_test` | 544 |

图像为 256×256 灰度 PNG，标签值 0、85、170、255 转为四类 one-hot。图像与标签一一对应；三个划分的病例没有重叠，代码沿用该划分。

**本机已有三个训练好的模型，可以直接推理，无需重新训练：**

```bash
python src/comp3710_lab2/part4.py --evaluate
python src/comp3710_lab2/part4_2.py --evaluate
python src/comp3710_lab2/part4_3.py --evaluate
```

去掉 `--evaluate` 会重新训练并保存到默认模型路径。`--epochs` 指定轮数；另一次实验应使用独立目录，例如 `--checkpoint results/part4_unet/demo/model.pt`，模型及其图像均保存在这个目录。默认训练轮数分别为 VAE 30、UNet 30、GAN 50；本次实际训练轮数见下表。

| 模型 | 默认模型文件 | 主要结果文件 | 本次训练与结果 |
|---|---|---|---|
| VAE | `results/part4_vae/model.pt` | `results/part4_vae/reconstructions.png`、`results/part4_vae/manifold.png` | CPU 30 轮，测试 BCE+KL/图像 1088.193 |
| UNet | `results/part4_unet/model.pt` | `results/part4_unet/segmentation.png`，终端输出逐类 DSC | A100 12 轮，测试四类 DSC 均 >0.9 |
| GAN | `results/part4_gan/model.pt` | `results/part4_gan/` 图像；`results/part4_gan/history.json` 损失历史 | A100 30 轮，约 283 秒 |

UNet 测试 DSC 按原标签 0/85/170/255 顺序为 **0.998847 / 0.937156 / 0.947372 / 0.971544**。VAE 损失不是准确率；GAN 图像已检查，最终真实度由演示教师评判。三个模型均已通过本机独立进程的模型加载和推理检查。完整记录见 [实验日志](docs/EXPERIMENT_LOG.md)。

## Rangpur 训练与结果

现场推理和取回图像的命令见 [展示入口](results/README.md#rangpur-现场运行)。集群代码和 UNet/GAN 已训练模型位于 `~/comp3710_lab2_20260910/`，结果使用与本机相同的实验目录。

如需重新训练，使用独立实验目录保存模型、图像和日志：

```bash
ssh rangpur.compute.eait.uq.edu.au
cd ~/comp3710_lab2_20260910
mkdir -p results/part4_unet/retrain

/opt/slurm/bin/sbatch --account=comp3710 --partition=comp3710 \
  --gres=gpu:a100:1 --cpus-per-task=4 --time=03:00:00 \
  --job-name=lab2-unet --output=results/part4_unet/retrain/training-%j.log \
  --wrap='"$HOME/miniconda3/envs/torch/bin/python" -u src/comp3710_lab2/part4_2.py --data-dir /home/groups/comp3710/OASIS --checkpoint results/part4_unet/retrain/model.pt'
```

VAE/GAN 对应脚本为 `part4.py` / `part4_3.py`，实验目录改为 `part4_vae` / `part4_gan`。单轮检查可改用 `--partition=a100-test --time=00:20:00`，在 Python 命令后加 `--epochs 1`。`a100-test` 每个作业最长 20 分钟；排队时间随资源变化。

## 讲义要求与当前状态

- Part 1 要求实际比较 GPU 张量 DFT、NumPy 朴素 DFT 和 NumPy FFT，并改变数据大小；没有 CUDA 时脚本会留下待测项。本次 A100 实测已完成，见 `results/part1/gpu_timing.log`。
- Part 2、Part 3.1 没有明确要求 GPU，本机 CPU 已运行。
- Part 3.2 要求准确率与速度测试，以及 Rangpur 现场推理和单轮训练；这些仍待完成。
- Part 4 三个模型已训练并保存证据；现场推理、结果解释和 GAN 真实度评判仍属于演示环节。Git 短课完成情况尚未提供。

课程还要求个人 GitHub 项目、相关提交记录和来源说明。此项目使用的仓库地址为 [tjrtjr/3702-demo2](https://github.com/tjrtjr/3702-demo2)。
