"""Part 4 Task 1: OASIS VAE and a two-dimensional latent manifold.

The lab specifies the task but supplies no VAE implementation. The encoder,
reparameterization, decoder and BCE + KL loss follow the PyTorch VAE example:
https://github.com/pytorch/examples/blob/main/vae/main.py
Necessary OASIS adaptations: grayscale images resized to 64x64 (4096 inputs),
two latent variables to plot the manifold directly, and the course's splits.
The example's 400-unit hidden layers and Adam learning rate 0.001 are retained.
"""

import argparse
from pathlib import Path
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from oasis import DEFAULT_DATA_ROOT, OASISDataset


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 64, 400)
        self.fc21 = nn.Linear(400, 2)
        self.fc22 = nn.Linear(400, 2)
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 64 * 64)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.reshape(-1, 64 * 64))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x.reshape(-1, 64 * 64), reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def resize(images, device):
    return F.interpolate(images.to(device), size=(64, 64), mode="bilinear",
                         align_corners=False, antialias=True)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = torch.zeros((), device=device)
    with torch.set_grad_enabled(optimizer is not None):
        for images in loader:
            images = resize(images, device)
            reconstruction, mu, logvar = model(images)
            loss = loss_function(reconstruction, images, mu, logvar)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.detach()
    return total_loss.item() / len(loader.dataset)


@torch.no_grad()
def visualize(model, dataset, device, checkpoint):
    model.eval()
    # 对均匀选取的测试图像显示原图和后验均值重建。
    indices = torch.linspace(0, len(dataset) - 1, 8).long().tolist()
    originals = resize(torch.stack([dataset[i] for i in indices]), device)
    mu, _ = model.encode(originals.flatten(1))
    reconstructed = model.decode(mu).view(-1, 1, 64, 64)
    save_image(torch.cat([originals, reconstructed]).cpu(),
               checkpoint.with_name(checkpoint.stem + "_reconstructions.png"), nrow=8)

    # 讲义允许直接采样展示流形：行/列分别改变两个潜变量。
    coordinates = torch.linspace(-2, 2, 10, device=device)
    zy, zx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    z = torch.stack([zx.flatten(), zy.flatten()], dim=1)
    manifold = model.decode(z).view(-1, 1, 64, 64)
    save_image(manifold.cpu(), checkpoint.with_name(checkpoint.stem + "_manifold.png"), nrow=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/part4.pt")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = VAE().to(device)
    workers = 2 if device.type == "cuda" else 0

    if not args.evaluate:
        train_loader = DataLoader(OASISDataset(args.data_dir, "train"), batch_size=128,
                                  shuffle=True, num_workers=workers)
        val_loader = DataLoader(OASISDataset(args.data_dir, "validate"), batch_size=128,
                                num_workers=workers)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        best_loss = float("inf")
        history = []
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        for epoch in range(args.epochs):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            val_loss = run_epoch(model, val_loader, device)
            history.append([train_loss, val_loss])
            print(f"Epoch {epoch + 1}/{args.epochs}, train loss/image={train_loss:.3f}, "
                  f"validation loss/image={val_loss:.3f}", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "validation_loss": val_loss, "history": history.copy()}, args.checkpoint)
        print(f"Training and validation time: {time.perf_counter() - start:.2f} s")

    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    print("Loaded epoch", saved["epoch"], "from", args.checkpoint)
    test_data = OASISDataset(args.data_dir, "test")
    test_loader = DataLoader(test_data, batch_size=128, num_workers=workers)
    print("Test loss/image:", run_epoch(model, test_loader, device))
    visualize(model, test_data, device, args.checkpoint)
    print("Saved test reconstructions and the 2D latent manifold beside the checkpoint.")


if __name__ == "__main__":
    main()
