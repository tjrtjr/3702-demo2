# Tutor 展示入口

依据《COMP3710 Lab Demonstration 2》Version 2.01。下面的展示内容对应讲义任务；实验详情见 [实验记录](../docs/EXPERIMENT_LOG.md)，代码来源见 [SOURCES.md](../SOURCES.md)。演示时还需要能回答 tutor 对代码、网络和结果的提问。

## 结果放在哪里

| 目录 | 先打开的文件 | 其余文件 |
|---|---|---|
| `part1/` | [GPU 计时](part1/gpu_timing.log) | 包含四种输入大小及正确性检查 |
| `part4_vae/` | [流形图](part4_vae/manifold.png)、[重建图](part4_vae/reconstructions.png) | `model.pt`、`metrics.json`、训练和推理日志 |
| `part4_unet/` | [分割对照图](part4_unet/segmentation.png)、[逐类 DSC](part4_unet/metrics.json) | `model.pt`、训练和推理日志 |
| `part4_gan/` | [生成脑图](part4_gan/generated.png)、[训练损失图](part4_gan/loss.png) | `model.pt`、`metrics.json`、原始损失和日志；逐轮图片统一放在 `epochs/` |

每个模型的权重、图和日志放在同一个实验目录。权重只在本机／Rangpur 保存，不随 GitHub 下载。整理保留了全部原始结果；历史日志里的旧保存路径没有改写。

## 按讲义展示的顺序

先打开 [个人 GitHub 仓库](https://github.com/tjrtjr/3702-demo2) 的代码、README、[提交记录](https://github.com/tjrtjr/3702-demo2/commits/main/) 和来源说明。**Part 4 必须有本人账号的项目及提交记录**，tutor 可能要求登录证明账号归属（讲义第 11 页）。

| 部分与讲义页码 | 展示什么、解释什么 | 当前情况 |
|---|---|---|
| Part 1，1 分，p2–5 | 方波与不同谐波数的重建、DFT 频谱；解释增加谐波后的变化及频谱差异。展示三个函数的 PyTorch 改写和显式 GPU DFT；展示三种算法在不同 N 下的计时、快慢顺序及原因。 | 本机代码已运行；A100 日志覆盖 N=256、512、1024、2048。 |
| Part 2，1 分，p5–9 | Eigenfaces、compactness 累计解释方差图、PCA→随机森林测试 accuracy 和 classification report；解释中心化、SVD、150 个主成分及分类过程。 | 上次测试 64.29%；随机森林未固定种子，再运行可能变化。 |
| Part 3.1，1 分，p9 | 两层 3×3、各 32 filters 的 CNN 与 dense 分类层、输入张量形状和测试分类表现；解释与 Part 2 的差别。 | 上次测试 81.06%。 |
| Part 3.2，4 分，p10 | CIFAR-10 ResNet-18 的准确率、训练时间和混合精度；**在 Rangpur 现场推理和训练一轮**。>90% 且训练较快占 1 分，现场运行占 1 分；94% 及与约 360 秒 V100 基准等效或更快的时间占 2 分。 | **尚未真实训练，无正式模型和达标证据。** |
| Part 4.1，1 分，p10 | 完成第二门短课 *Version Control for Teams using Git*，展示课程完成状态。 | **完成情况尚未确认。** |
| Part 4 VAE，p11 | 展示 OASIS 训练结果及 **manifold 流形图**；解释编码器、采样、解码器和损失。重建图用于辅助解释。 | 已训练 30 轮，有模型、流形和日志。 |
| Part 4 UNet，p12 | **现场在 test set 推理**，展示 MRI、真值、预测分割及每个标签 DSC；解释 categorical／one-hot、skip connections、训练与验证方法。 | 已训练；四类 DSC 均 >0.9。 |
| Part 4 GAN，p12 | 展示逼真且不同的 OASIS 脑图，以及训练证据（生成图、loss 图等）；解释 G/D 和 mode collapse。 | 已训练 30 轮；最终真实度由 tutor 判断。 |

Part 4 的三项任务逐级累计：只做 VAE 上限 3/7 分，VAE+UNet 上限 5/7 分，三项全部完成上限 7/7 分；Git 短课另计 1 分。代码功能、解释、GitHub 文档／提交及代码组织也在评分范围内。

讲义明确要求现场运行的是 **Part 3.2 的 Rangpur 推理和一轮训练，以及 UNet 的测试集推理**。其他部分仍需展示和解释，但没有要求现场重新完成全部训练。Part 3.1 没有额外要求 accuracy/loss 曲线；VAE 没有要求必须使用 UMAP。

## 本机展示 Part 1、2、3.1 和已有图像

在项目根目录打开终端：

```bash
source .venv/bin/activate
export SCIKIT_LEARN_DATA="$PWD/data"
export OMP_NUM_THREADS=4

python src/comp3710_lab2/part1.py
python src/comp3710_lab2/part2.py
python src/comp3710_lab2/part3.py
```

逐条运行。Part 1 的图窗展示方波重建和频谱，同时打开上面的 GPU 计时日志；本机没有 CUDA，不会重新测出 GPU 时间。Part 2 先展示 eigenfaces，关闭图窗后看 compactness，再关闭后看终端分类报告。Part 3.1 当前脚本会训练 20 轮再输出测试结果，可以提前运行并保留终端输出。

本机检查已保存的 Part 4 模型：

```bash
python src/comp3710_lab2/part4.py --evaluate
python src/comp3710_lab2/part4_2.py --evaluate
python src/comp3710_lab2/part4_3.py --evaluate
```

这些命令加载模型并更新对应目录中的图，不重新训练。VAE 流形图的行／列分别改变两个潜变量；重建图上排为原图、下排为重建。UNet 图的三列是原 MRI、真值、预测。GAN 的 [真实训练图](part4_gan/real_training.png)、[第 1 轮](part4_gan/epochs/epoch_001.png)、[第 30 轮](part4_gan/epochs/epoch_030.png) 可用于说明训练变化；不能仅凭 loss 判定图像逼真或排除 mode collapse。

## Rangpur 现场运行

演示前登录并申请 GPU；等分配成功后，在 tutor 面前执行后面的推理命令。排队时间没有保证。

```bash
ssh rangpur.compute.eait.uq.edu.au
cd ~/comp3710_lab2_20260910
/opt/slurm/bin/srun --account=comp3710 --partition=a100-test \
  --gres=gpu:a100:1 --cpus-per-task=4 --time=00:20:00 --pty bash
```

获得 GPU 后运行 UNet：

```bash
"$HOME/miniconda3/envs/torch/bin/python" -u src/comp3710_lab2/part4_2.py \
  --data-dir /home/groups/comp3710/OASIS --evaluate
```

展示终端的 `Test images: 544`、四个标签的 DSC 和 `Every label DSC > 0.9: True`，再展示生成的 `results/part4_unet/segmentation.png`。已有 A100 实测按标签 0/85/170/255 顺序为 **0.998847 / 0.937156 / 0.947372 / 0.971544**；要求是每一类 >0.9。

在本机另开终端，从项目根目录取回刚刚生成的图并打开：

```bash
scp rangpur.compute.eait.uq.edu.au:~/comp3710_lab2_20260910/results/part4_unet/segmentation.png results/part4_unet/
open results/part4_unet/segmentation.png
```

**Part 3.2 需要先完成正式训练并准备好数据和模型，当前不能直接完成下面的推理。** 准备好 `results/part3_2/model.pt` 后，在同一 GPU 会话中执行：

```bash
"$HOME/miniconda3/envs/torch/bin/python" -u src/comp3710_lab2/part3_2.py --evaluate
"$HOME/miniconda3/envs/torch/bin/python" -u src/comp3710_lab2/part3_2.py \
  --epochs 1 --checkpoint results/part3_2/demo/model.pt
```

单轮训练使用独立目录。演示结束后输入 `exit` 释放 GPU 会话，再输入 `exit` 退出 SSH。
