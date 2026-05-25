"""Train a tiny MLP autoencoder on normal vibration windows (Chapter 11)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent


class TinyAutoencoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def windowize(signal: np.ndarray, win: int, stride: int) -> np.ndarray:
    """signal: (T, C) → (n, win*C) — flattened windows."""
    n = (signal.shape[0] - win) // stride + 1
    windows = np.stack([signal[i * stride: i * stride + win] for i in range(n)], axis=0)
    return windows.reshape(n, -1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(cfg["seed"])

    data_path = ROOT.parent.parent / "datasets" / "vibration_normal.npy"
    if not data_path.exists():
        raise SystemExit(f"missing {data_path}; run generate_synthetic.py first")
    signal = np.load(data_path)
    print(f"loaded normal signal: {signal.shape} @ {cfg['sample_rate_hz']} Hz")

    # Split first 80% train, last 20% held-out normal (for threshold calibration)
    split = int(0.8 * len(signal))
    train_sig = signal[:split]
    val_sig = signal[split:]

    win = cfg["window_samples"]
    stride = cfg["stride_samples"]
    train_w = windowize(train_sig, win, stride)
    val_w = windowize(val_sig, win, stride)
    in_dim = win * cfg["n_channels"]
    print(f"train windows: {train_w.shape}   val windows: {val_w.shape}")

    train_ds = TensorDataset(torch.from_numpy(train_w))
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    model = TinyAutoencoder(in_dim=in_dim, latent_dim=cfg["latent_dim"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.MSELoss()

    history = []
    model.train()
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        for (x,) in train_dl:
            opt.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(train_ds)
        history.append(epoch_loss)
        print(f"epoch {epoch + 1:>2}/{cfg['epochs']}  train_mse={epoch_loss:.5f}")

    # Threshold from held-out normal windows
    model.eval()
    with torch.no_grad():
        val_x = torch.from_numpy(val_w)
        val_recon = model(val_x)
        per_window_err = ((val_x - val_recon) ** 2).mean(dim=1).cpu().numpy()
    threshold = float(np.quantile(per_window_err, cfg["threshold_quantile"]))
    print(f"\nthreshold ({cfg['threshold_quantile']:.0%} of held-out normal error) = {threshold:.5f}")

    # Save
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg,
        "threshold": threshold,
        "in_dim": in_dim,
    }, ROOT / "results" / "autoencoder.pt")
    print(f"saved {ROOT / 'results' / 'autoencoder.pt'}")

    # Loss plot
    plt.figure(figsize=(7, 3))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("autoencoder training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = ROOT / "results" / "training_loss.png"
    plt.savefig(out_png, dpi=120)
    print(f"saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
