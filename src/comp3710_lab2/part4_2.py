"""Part 4, Task 2: OASIS U-Net segmentation and test-set inference."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from oasis import DEFAULT_DATA_ROOT, OASISDataset


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four-level encoder/decoder with the original U-Net skip connections."""

    def __init__(self):
        super().__init__()
        # 讲义指定 U-Net，但未提供代码。架构依据讲义所链接的原论文：
        # https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
        # 为适配 256x256 OASIS 切片，使用 padding=1 保持输出尺寸；
        # 基础通道数取 32（原论文为 64），其余各层按原结构逐级加倍。
        self.enc1 = double_conv(1, 32)
        self.enc2 = double_conv(32, 64)
        self.enc3 = double_conv(64, 128)
        self.enc4 = double_conv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottom = double_conv(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = double_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = double_conv(64, 32)
        self.output = nn.Conv2d(32, 4, 1)

    def forward(self, images):
        e1 = self.enc1(images)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottom(self.pool(e4))
        x = self.dec4(torch.cat((self.up4(x), e4), dim=1))
        x = self.dec3(torch.cat((self.up3(x), e3), dim=1))
        x = self.dec2(torch.cat((self.up2(x), e2), dim=1))
        x = self.dec1(torch.cat((self.up1(x), e1), dim=1))
        # 四类 logits 用于 categorical cross entropy；推理时沿类别维 softmax。
        return self.output(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    intersection = torch.zeros(4, dtype=torch.float64, device=device)
    total = torch.zeros(4, dtype=torch.float64, device=device)
    examples = None
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        probabilities = logits.float().softmax(dim=1)
        prediction = probabilities.argmax(dim=1)
        truth = targets.argmax(dim=1)
        for label in range(4):
            predicted_label = prediction == label
            true_label = truth == label
            intersection[label] += (predicted_label & true_label).sum()
            total[label] += predicted_label.sum() + true_label.sum()
        if examples is None:
            examples = (images[:4].cpu(), truth[:4].cpu(), prediction[:4].cpu())
    # 每类 DSC 在整个 split 上累积像素后计算；无真值且无预测时记为 NaN。
    dice = torch.where(total > 0, 2 * intersection / total, torch.nan)
    return dice.cpu(), examples


def save_segmentation(examples, path):
    images, truth, prediction = examples
    figure, axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)), squeeze=False)
    for row in range(len(images)):
        axes[row, 0].imshow(images[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(truth[row], cmap="viridis", vmin=0, vmax=3)
        axes[row, 2].imshow(prediction[row], cmap="viridis", vmin=0, vmax=3)
        for column in range(3):
            axes[row, column].axis("off")
    for column, title in enumerate(("OASIS image", "Ground truth", "U-Net prediction")):
        axes[0, column].set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/part4_2.pt"))
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(), flush=True)
    model = UNet().to(device)
    # 讲义未指定训练参数：batch size=8，Adam lr=0.001，默认 30 个 epoch。
    loader_options = dict(batch_size=8, num_workers=4 if device.type == "cuda" else 0,
                          pin_memory=device.type == "cuda")
    if not args.evaluate:
        train_data = OASISDataset(args.data_dir, "train", segmentation=True)
        validation_data = OASISDataset(args.data_dir, "validate", segmentation=True)
        train_loader = DataLoader(train_data, shuffle=True, **loader_options)
        validation_loader = DataLoader(validation_data, **loader_options)
        print(f"Train: {len(train_data)}, validation: {len(validation_data)}", flush=True)
        # CrossEntropyLoss 直接接收四通道 one-hot target，内部完成 log-softmax。
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_minimum_dice = -1.0
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * images.size(0)
            validation_dice, _ = evaluate(model, validation_loader, device)
            print(f"Epoch {epoch + 1}/{args.epochs}, loss: {total_loss / len(train_data):.4f}, "
                  f"validation DSC (0/85/170/255): {validation_dice.tolist()}", flush=True)
            # 以验证集最弱类别的 DSC 选模型，测试集只用于训练结束后的推理。
            if validation_dice.min().item() > best_minimum_dice:
                best_minimum_dice = validation_dice.min().item()
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                            "validation_dice": validation_dice.tolist()}, args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded:", args.checkpoint, "epoch:", checkpoint["epoch"], flush=True)
    test_data = OASISDataset(args.data_dir, "test", segmentation=True)
    test_loader = DataLoader(test_data, **loader_options)
    test_dice, examples = evaluate(model, test_loader, device)
    print("Test images:", len(test_data), flush=True)
    for label, value in enumerate((0, 85, 170, 255)):
        print(f"Label {label} (mask value {value}): test DSC = {test_dice[label]:.6f}")
    print("Every label DSC > 0.9:", bool((test_dice > 0.9).all()), flush=True)
    figure_path = args.checkpoint.with_suffix(".png")
    save_segmentation(examples, figure_path)
    print("Saved segmentation examples:", figure_path, flush=True)


if __name__ == "__main__":
    main()
