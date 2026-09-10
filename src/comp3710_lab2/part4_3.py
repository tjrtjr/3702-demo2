"""Part 4 Task 3：用课程 OASIS 数据训练 DCGAN 并展示生成结果。

讲义没有提供 GAN 完整代码；以下网络、初始化和训练步骤改编自
PyTorch 官方 DCGAN 教程（Nathan Inkawhich）：
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py

沿用教程的 64×64、100 维噪声、64 基础通道数、BCE 和 Adam 参数。
适配 OASIS 时改为单通道，将 256×256 切片缩至 64×64 并归一化到 [-1, 1]。
batch_size=64、50 个 epoch 是讲义未指定的训练选择；不使用预训练模型。
生成图和损失曲线用于检查训练，不能单凭损失断言真实度或排除模式崩溃。
"""

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from oasis import DEFAULT_DATA_ROOT, OASISDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SIZE = 64
LATENT_SIZE = 100
BATCH_SIZE = 64


def weights_init(layer):
    # 官方教程：卷积 N(0, 0.02)，BatchNorm 权重 N(1, 0.02)、偏置为 0。
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, 0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.zeros_(layer.bias)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_SIZE, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.main(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.main(images).view(-1)


def prepare_images(images, device):
    images = F.interpolate(
        images.to(device), size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False, antialias=True,
    )
    return images * 2 - 1


def train_batch(generator, discriminator, real, optimizer_g, optimizer_d, criterion):
    # 官方教程步骤 1：分别用真图、detach 后的假图累积 D 的梯度。
    discriminator.zero_grad()
    labels = torch.ones(real.size(0), device=real.device)
    loss_real = criterion(discriminator(real), labels)
    loss_real.backward()

    noise = torch.randn(real.size(0), LATENT_SIZE, 1, 1, device=real.device)
    fake = generator(noise)
    labels.fill_(0)
    loss_fake = criterion(discriminator(fake.detach()), labels)
    loss_fake.backward()
    optimizer_d.step()

    # 官方教程步骤 2：将同一批假图的目标设为真，更新 G。
    generator.zero_grad()
    labels.fill_(1)
    loss_g = criterion(discriminator(fake), labels)
    loss_g.backward()
    optimizer_g.step()
    return loss_g.item(), (loss_real + loss_fake).item()


@torch.inference_mode()
def save_generated(generator, noise, path):
    generator.eval()
    # 使用统一 [-1, 1] → [0, 1] 映射，保留不同生成图的强度差异。
    save_image((generator(noise).cpu() + 1) / 2, path, nrow=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "results" / "part4_gan" / "model.pt")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; OASIS training resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device))
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    figure_dir = args.checkpoint.parent

    if args.evaluate:
        generator = Generator().to(device)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        generator.load_state_dict(saved["generator"])
        # 每次推理使用新噪声；训练中的 fixed_noise 仅用于比较同一位置的演化。
        noise = torch.randn(64, LATENT_SIZE, 1, 1, device=device)
        path = figure_dir / "generated.png"
        save_generated(generator, noise, path)
        print(f"Loaded {args.checkpoint} ({saved['epochs']} epochs); saved {path}")
        return

    epoch_dir = figure_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(999)
    train_data = OASISDataset(args.data_dir, "train")
    workers = 2 if device.type == "cuda" else 0
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=workers, persistent_workers=workers > 0)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_SIZE, 1, 1, device=device)
    history = {"generator": [], "discriminator": []}
    print(f"Training on {len(train_data)} OASIS slices; {len(loader)} batches/epoch")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        for batch_index, images in enumerate(loader):
            real = prepare_images(images, device)
            if epoch == 0 and batch_index == 0:
                save_image((real.cpu() + 1) / 2, figure_dir / "real_training.png", nrow=8)
            loss_g, loss_d = train_batch(
                generator, discriminator, real, optimizer_g, optimizer_d, criterion,
            )
            history["generator"].append(loss_g)
            history["discriminator"].append(loss_d)
            if batch_index % 50 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, batch {batch_index + 1}/{len(loader)}: "
                      f"G={loss_g:.4f}, D={loss_d:.4f}", flush=True)

        save_generated(generator, fixed_noise, epoch_dir / f"epoch_{epoch + 1:03d}.png")
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start
        torch.save({"generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "epochs": epoch + 1, "image_size": IMAGE_SIZE,
                    "fixed_noise": fixed_noise.cpu(), "history": history,
                    "training_seconds": training_seconds}, args.checkpoint)
        (args.checkpoint.parent / "history.json").write_text(json.dumps(history), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["generator"], label="G")
        ax.plot(history["discriminator"], label="D")
        ax.set(xlabel="Training iteration", ylabel="BCE loss",
               title="OASIS DCGAN training loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "loss.png", dpi=150)
        plt.close(fig)
        print(f"Finished epoch {epoch + 1}; elapsed {training_seconds:.1f} s; "
              f"saved {args.checkpoint}", flush=True)

    noise = torch.randn(64, LATENT_SIZE, 1, 1, device=device)
    save_generated(generator, noise, figure_dir / "generated.png")
    print(f"Saved training evidence to {figure_dir}")


if __name__ == "__main__":
    main()
