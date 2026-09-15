# Tutor 展示入口

依据《COMP3710 Lab Demonstration 2》Version 2.01，2026-09-15 重新对照 PDF、代码和实际日志核查。**明确的算法要求已覆盖，但代码并非全部是 PDF 原码，保存的结果也并非每个都是必需展示项。** 下文分别说明。代码来源见 [SOURCES.md](../SOURCES.md)，实测过程见 [实验记录](../docs/EXPERIMENT_LOG.md)。

## Demo 按这个顺序展示

**GitHub／Git 短课 → Part 1 → Part 2 → Part 3.1 → Part 3.2 → VAE → UNet → GAN。** 这是便于讲解的建议顺序；PDF 没有规定演示顺序。展示仓库、终端和现有图片即可，PDF 没有要求制作 PPT 或 notebook。

| 顺序、讲义位置 | 打开什么／做什么 | 需要能解释的内容 |
|---|---|---|
| 0. GitHub 与 Git 短课，p10–11 | 打开[本人仓库](https://github.com/tjrtjr/3702-demo2)、README、[提交记录](https://github.com/tjrtjr/3702-demo2/commits/main/) 和来源说明。展示 *Version Control for Teams using Git* 完成状态。 | Part 4 必须有本人账号的仓库和相关提交，tutor 可能要求登录。**短课完成情况目前尚未确认**，不能用仓库代替课程完成证明。 |
| 1. Part 1，p2–5 | 运行 `part1.py`，展示方波／不同谐波数、DFT 频谱；打开 [GPU 计时表](part1/gpu_timing.log)。 | 谐波增多后的变化、频谱差异；三个函数怎样改成 PyTorch；显式 GPU DFT、NumPy DFT、NumPy FFT 随 N 变化的速度与原因。已有 N=256/512/1024/2048 的 GPU 实测。 |
| 2. Part 2，p5–9 | 运行 `part2.py`，依次展示 eigenfaces → compactness → RF 测试 accuracy／classification report。 | 中心化、SVD、150 个 PCA 特征，再交给随机森林。上次 accuracy **64.29%**；讲义没有固定 RF 随机种子，重跑可能变化。 |
| 3. Part 3.1，p9 | 打开 [CNN 代码](../src/comp3710_lab2/part3.py) 和运行后的终端：输入形状、网络、测试 accuracy。 | 两层卷积均为 3×3、32 filters，dense 分类；PCA 提取固定线性特征，CNN 联合学习特征和分类。上次 accuracy **81.06%**。 |
| 4. Part 3.2，p10 | 打开 [35 轮日志](part3_2/training.log)／[指标](part3_2/metrics.json)，再在 **Rangpur 当场推理并训练一轮**，命令见下文。 | 手写残差块、附录 B 参考来源、混合精度、准确率及计时范围。正式模型 **94.19%，228.95 秒，A100**；单轮预演的 43.14% 不是正式成绩。 |
| 5. Part 4 VAE，p11 | 打开 [manifold](part4_vae/manifold.png) 和 [训练日志](part4_vae/training.log)。 | 编码器、重参数采样、解码器、BCE+KL；图的行／列分别改变两个潜变量。已训练 30 轮。重建图可在被问到时补充。 |
| 6. Part 4 UNet，p12 | **当场加载模型，在 test set 推理**；展示终端逐类 DSC 和 [MRI／真值／预测图](part4_unet/segmentation.png)。 | 四类 categorical／one-hot、skip connections、训练／验证／测试划分。四类 DSC **0.998847 / 0.937156 / 0.947372 / 0.971544**，均 >0.9。 |
| 7. Part 4 GAN，p12 | 打开 [生成脑图](part4_gan/generated.png)、[训练 loss](part4_gan/loss.png)，必要时打开 [训练日志](part4_gan/training.log)。 | G／D 怎样训练，图像是否像不同的脑、什么是 mode collapse。已训练 30 轮；最终真实度由 tutor 判断，不能只凭 loss 宣称已完全排除 mode collapse。 |

**必须当场执行的操作是 Part 3.2 在 Rangpur 推理＋训练一轮，以及 UNet 的测试集推理。** 提前保存的日志不能代替这两项现场操作。其余部分需要展示并解释，但 PDF 没有要求现场重新跑完全部训练；Part 3.1 可以提前运行好，保留终端。

## 先准备本机 Part 1、2、3.1

在本机终端设置环境：

```bash
cd "/Users/junrongtang/Desktop/3710 demo2"
source .venv/bin/activate
export SCIKIT_LEARN_DATA="$PWD/data"
export OMP_NUM_THREADS=4
```

按展示顺序逐条运行，等上一部分结束再执行下一条：

```bash
python src/comp3710_lab2/part1.py
python src/comp3710_lab2/part2.py
python src/comp3710_lab2/part3.py
```

- Part 1：图窗展示完后关闭，让脚本继续；另打开 GPU 计时日志。本机没有 CUDA，GPU 数据来自已经完成的 A100 实验。
- Part 2：先展示 eigenfaces 并关闭图窗 → 展示 compactness 并关闭 → 最后看终端分类报告。前面的图窗不关闭，后面的步骤不会继续。
- Part 3.1：每次启动都会训练 20 轮，没有加载已训练模型的模式。可以 demo 前在单独终端运行好；无需逐行讲每轮 loss 或每条预测。

模型权重和数据保存在本机／Rangpur，不在 GitHub 中。从 GitHub 新下载仓库不能直接加载这些模型；当天用现在已有的项目目录。

## Rangpur 现场运行

提前准备 SSH 登录，在准备展示 Part 3.2 时申请 GPU。下面申请的会话时限是 **20 分钟，从资源分配成功开始计时**；不要分配好后放置太久。排队时长没有保证，出现 GPU 节点的 shell 提示符后再执行 Python 命令。

```bash
ssh rangpur.compute.eait.uq.edu.au
cd ~/comp3710_lab2_20260910
/opt/slurm/bin/srun --account=comp3710 --partition=a100-test \
  --gres=gpu:a100:1 --cpus-per-task=4 --time=00:20:00 --pty bash
export OMP_NUM_THREADS=4
```

### 第 4 步：Part 3.2 正式模型推理，再训练一轮

在获得的 GPU 会话内逐条执行：

```bash
"$HOME/miniconda3/envs/torch/bin/python" -u src/comp3710_lab2/part3_2.py --evaluate
"$HOME/miniconda3/envs/torch/bin/python" -u src/comp3710_lab2/part3_2.py \
  --epochs 1 --checkpoint results/part3_2/demo/model.pt
```

第一条加载正式模型，已有结果为 `Loaded ... (35 training epochs)` 和 `Test accuracy (10000 images): 94.1900%`。第二条从头训练一轮，展示一轮训练确实能运行；它写入 `demo/model.pt`，不会覆盖正式模型。

打开原 [完整训练日志](part3_2/training.log) 中的 `Training time: 228.95 s`。计时包含训练循环的数据加载、增强及 CUDA 同步，不含排队、下载、模型初始化、测试和保存。这是 **A100 实测**，不要说成 V100 实测。PDF 的分项是 >90% 且训练较快（1 分）、现场操作（1 分）、混合精度下 94% 与约 360 秒 V100 基准等效或更快（2 分）；已测得的数值和硬件如实展示，由 tutor 判定。

### 第 5 步：切回本机，展示 VAE

打开 [manifold.png](part4_vae/manifold.png)，解释两个潜变量变化时解码出的脑图；用 [训练日志](part4_vae/training.log) 证明完成训练。需要解释输入／重建关系时再打开 [reconstructions.png](part4_vae/reconstructions.png)。这里无需运行训练或重新推理，GPU 会话保留给下一步。

### 第 6 步：同一 GPU 会话运行 UNet

```bash
"$HOME/miniconda3/envs/torch/bin/python" -u src/comp3710_lab2/part4_2.py \
  --data-dir /home/groups/comp3710/OASIS --evaluate
```

展示终端的 `Test images: 544`、四个标签的 DSC 和 `Every label DSC > 0.9: True`。已有四类 DSC 按标签 0/85/170/255 顺序为 **0.998847 / 0.937156 / 0.947372 / 0.971544**。这是全部测试图像累计像素的逐类 DSC，不是每张 MRI 都有 >0.9 的保证。

在**本机另一个终端**取回刚刚生成的图并打开：

```bash
cd "/Users/junrongtang/Desktop/3710 demo2"
scp rangpur.compute.eait.uq.edu.au:~/comp3710_lab2_20260910/results/part4_unet/segmentation.png results/part4_unet/
open results/part4_unet/segmentation.png
```

三列依次是原 MRI、真值、预测。展示完后回到远程终端输入 `exit` 释放 GPU，再输入 `exit` 退出 SSH。

### 第 7 步：本机展示 GAN

打开 [generated.png](part4_gan/generated.png) 和 [loss.png](part4_gan/loss.png)。需要对比时再打开 [真实训练图](part4_gan/real_training.png) 或 [第 1 轮](part4_gan/epochs/epoch_001.png)／[第 30 轮](part4_gan/epochs/epoch_030.png)。无需现场重新训练 GAN，也无需遍历 30 张逐轮图。

## 核查：代码是否全部来自 PDF

**不是全部逐行来自 PDF。** 讲义给了代码的部分沿用其实现；要求自行实现、但没给完整代码的部分补齐了网络和运行细节。没有调用预制整网或加载外部预训练权重。具体边界如下：

| 部分 | 讲义代码／明确要求 | PDF 没有给出、当前补齐的内容 |
|---|---|---|
| Part 1 | NumPy 的三个核心函数沿用讲义；20/50 项谐波、PyTorch 改写和多种 N 的 GPU DFT／NumPy DFT／FFT 比较都是题目要求。 | PyTorch 张量实现；计时采用预热、同步和重复平均，额外做 PyTorch 数值一致性检查。 |
| Part 2 | 数据、划分、SVD 150 维、eigenfaces、compactness、RF 参数和分类报告均沿用讲义。 | 修正转录／语法错误及代码格式。没有另做 PCA 维数扫描、原始像素 RF 或交叉验证。 |
| Part 3.1 | 沿用 LFW 预处理，满足两层 3×3／32 filters、dense 分类，使用允许的 Adam／交叉熵。 | PDF 没有完整模型；ReLU、池化、dense128、20 轮、batch32、lr=0.001、seed42 是实现选择。每轮训练输出是辅助观察。 |
| Part 3.2 | 移植 PDF 附录 B 链接的课程 JAX 代码，沿用通道数、35 轮、batch128、SGD、学习率和增强方式；AMP 对应题目要求。 | PyTorch 移植、数据加载、模型保存／加载和演示命令。属于课程参考移植，不能称 PDF 原码；35 轮来自参考实现，不是 PDF 硬性轮数要求。 |
| Part 4 | PDF 指定 OASIS VAE／UNet／GAN 的任务和指标，**没有提供三种模型的完整代码**。 | VAE 与 DCGAN 使用已注明的官方 PyTorch 示例；UNet 按讲义链接的架构实现。输入尺寸、宽度、优化器和训练轮数等是实现选择，详见来源说明。本次实际 VAE 30 轮、UNet 12 轮、GAN 30 轮，PDF 未规定这些轮数。 |

## 核查：有没有多跑、多存东西

**有额外验证、重复推理和辅助输出；它们不是独立的必做评分任务。** 这次核查没有重新训练，也没有删除已有结果。

| 内容 | 是否必须／demo 怎么处理 |
|---|---|
| Part 1 谐波、频谱、不同 N 的三算法 GPU 比较 | 讲义要求，保留并展示。NumPy DFT 与 FFT 的 `allclose` 也是讲义原代码。 |
| Part 1 的 PyTorch `allclose`、预热、FFT 100 次／GPU DFT 3 次平均 | 辅助正确性和计时检查；重复的是同一方法的测量，没有新增算法。不必逐项展示检查输出。 |
| Part 2 的 predictions、which-correct、accuracy、classification report | 都是 PDF 给出的输出，不是私加实验。演示重点放在总体分类表现。 |
| Part 3.1 每轮 loss／train accuracy、classification report | 辅助输出；PDF 没有要求额外画 accuracy/loss 曲线，当前也没跑曲线实验。 |
| Part 3.2 提前跑的一轮训练及 `demo/` 日志 | 是事前预演；当场再训练一轮才满足现场要求。它不是另一个需要达到 94% 的完整实验。 |
| VAE 的 manifold | 明确必需。当前使用二维潜变量采样，属于 PDF 允许的方法，**没有必要再跑 UMAP**。 |
| VAE 的重建图、测试 loss、独立推理日志 | 辅助理解和模型验证，不是 PDF 单独要求的展示项。测试 loss 不是 accuracy。 |
| UNet 的逐类 DSC、分割图和现场 test 推理 | 明确必需；不能只展示训练集图或一个平均 DSC。 |
| GAN 的最终脑图及训练证据 | 明确必需；生成图＋loss 图是当前采用的证据形式。是否足够逼真、不同，仍需 tutor 判断。 |
| GAN 的全部 `epochs/`、`generated_gpu.png`、真实图及 CPU 重载新图 | 辅助保存／验证；30 张逐轮图来自同一次训练，不是 30 次独立实验。只需优先展示最终图和 loss。 |
| 模型重载、CPU／GPU 检查、数据配对／完整性验证、早期单批前向／反向试跑 | 都是额外运行检查，不是额外评分课题，也不需要在 demo 逐项展示。 |

Part 4 按难度累计：**只做 VAE 上限 3/7，VAE＋UNet 上限 5/7，三项全做上限 7/7**；Git 短课另计 1 分。现在三种模型都做了，是覆盖最高难度的任务范围，并不是 PDF 要求每位学生都必须跑三种。仍须确认 Git 短课完成状态，并在 demo 完成现场操作、解释代码和结果；现有结果不代表已经取得对应分数。

所有模型、日志及辅助图片仍按实验目录保留。历史日志中的旧保存路径没有改写；仓库中的来源和 AI 使用记录也保留，演示时应如实说明。
