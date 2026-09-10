"""Part 3.2: PyTorch port of the DAWNBench reference linked in Appendix B.

Sources (Shakes' jax-vision, Apache-2.0):
https://github.com/shakes76/jax-vision/blob/main/networks.py
https://github.com/shakes76/jax-vision/blob/main/preprocess.py
https://github.com/shakes76/jax-vision/blob/main/test_resnet_cifar.py

Keep the reference's 64/128/256/256 channels and projection in every block,
including its narrower final stage. These are the course reference's ResNet-18
variant. The 35 epochs, batch size, augmentation, SGD and learning-rate schedule
come from that reference. CUDA automatic mixed precision is required by Part 3.2.
PyTorch DataLoader visits every training image once per epoch (including the
last partial batch); the reference instead batches a repeated TF dataset.
"""

import argparse
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 35
BATCH_SIZE = 128
CIFAR_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR_STD = (0.24703223, 0.24348513, 0.26158784)


class ResNetBlock(nn.Module):
    """Two 3x3 convolutions plus the reference's 1x1 projection shortcut."""

    def __init__(self, in_channels, channels, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride,
            padding=1 if stride == 1 else 0, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1),
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        # Haiku SAME padding for the even-sized CIFAR feature maps at stride 2.
        out = F.pad(x, (0, 1, 0, 1)) if self.stride == 2 else x
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + shortcut)


class ResNet18(nn.Module):
    """The manually implemented CIFAR ResNet-18 used in the course reference."""

    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResNetBlock(64, 64, 1), ResNetBlock(64, 64, 1),
            ResNetBlock(64, 128, 2), ResNetBlock(128, 128, 1),
            ResNetBlock(128, 256, 2), ResNetBlock(256, 256, 1),
            ResNetBlock(256, 256, 2), ResNetBlock(256, 256, 1),
        )
        self.classifier = nn.Linear(256, 10)
        # Haiku Conv2D defaults: truncated normal, std = 1 / sqrt(fan_in).
        # https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/conv.py
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                std = 1 / math.sqrt(layer.weight[0].numel())
                nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
        # The reference explicitly initializes the output weights to zero.
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.blocks(self.initial(x))
        return self.classifier(x.mean(dim=(2, 3)))


def learning_rate(step, total_steps):
    """The reference's optax.linear_onecycle_schedule, with fixed momentum.

    https://github.com/google-deepmind/optax/blob/main/optax/schedules/_schedule.py
    """
    start = int(total_steps * (15 / EPOCHS))
    final = int(total_steps * (30 / EPOCHS))
    if step < start:
        return 0.005 + (0.1 - 0.005) * step / start
    if step < final:
        return 0.1 + (0.005 - 0.1) * (step - start) / (final - start)
    fraction = min((step - final) / (total_steps - final), 1.0)
    return 0.005 + (0.005 / 200 - 0.005) * fraction


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, scaler, device, epoch, total_steps):
    model.train()
    loss_sum = torch.zeros((), device=device)
    total = 0
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        step = epoch * len(loader) + batch_index
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, total_steps)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = F.cross_entropy(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach() * labels.size(0)
        total += labels.size(0)
    return loss_sum.item() / total


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum()
        total += labels.size(0)
    assert total == len(loader.dataset)
    return correct.item() / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Train this many epochs of the reference's 35-epoch schedule.")
    parser.add_argument("--evaluate", action="store_true", help="Run checkpoint inference only.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to save or load.")
    args = parser.parse_args()
    if not 1 <= args.epochs <= EPOCHS:
        parser.error("--epochs must be between 1 and 35")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 2 if device.type == "cuda" else 0
    checkpoint = args.checkpoint or PROJECT_ROOT / "results" / "part3_2" / (
        "demo/model.pt" if args.epochs < EPOCHS and not args.evaluate else "model.pt"
    )
    torch.manual_seed(42)
    print(f"Device: {device}; mixed precision: {device.type == 'cuda'}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device))

    normalize = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    test_set = datasets.CIFAR10(PROJECT_ROOT / "data", train=False, download=True,
                               transform=transforms.Compose(normalize))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=workers)
    model = ResNet18().to(device)

    if args.evaluate:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        print(f"Loaded {checkpoint} ({saved['epochs']} training epochs)")
    else:
        augmentation = transforms.Compose(normalize + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        ])
        train_set = datasets.CIFAR10(PROJECT_ROOT / "data", train=True, download=True,
                                    transform=augmentation)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=workers, persistent_workers=workers > 0)
        # SGD weight_decay has the same gradient as 0.5 * 5e-4 * sum(p**2)
        # in the reference loss, applied to all parameters, including BN/bias.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                    momentum=0.9, weight_decay=5e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        epochs = args.epochs
        total_steps = EPOCHS * len(train_loader)
        synchronize(device)
        start = time.perf_counter()
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            loss = train_epoch(model, train_loader, optimizer, scaler,
                               device, epoch, total_steps)
            synchronize(device)
            print(f"Epoch {epoch + 1}/{epochs}: cross entropy={loss:.5f}, "
                  f"time={time.perf_counter() - epoch_start:.2f} s", flush=True)
        training_seconds = time.perf_counter() - start
        print(f"Training time: {training_seconds:.2f} s ({training_seconds / 60:.2f} min)")

    synchronize(device)
    start = time.perf_counter()
    accuracy = evaluate(model, test_loader, device)
    synchronize(device)
    print(f"Test accuracy ({len(test_set)} images): {accuracy:.4%}")
    print(f"Inference time: {time.perf_counter() - start:.2f} s")
    if not args.evaluate:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epochs": epochs,
                    "test_accuracy": accuracy, "training_seconds": training_seconds}, checkpoint)
        print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
