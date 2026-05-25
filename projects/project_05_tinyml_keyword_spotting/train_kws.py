"""Train a small DS-CNN-style keyword spotter on the synthetic dataset (Ch 12).

Pipeline:
    raw audio (1 s @ 16 kHz)
        → MFCC features (~49 frames × 10 coeffs)
        → small DS-CNN (~25-50 K params)
        → softmax over n_classes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent


# -------------------- A minimal MFCC implementation in NumPy --------------------

def compute_mfcc(signal: np.ndarray, rate: int, win_ms: int, hop_ms: int,
                 n_mfcc: int, n_fft: int = 512, n_mel: int = 40) -> np.ndarray:
    """Very simplified MFCC: STFT → mel filterbank → log → DCT.
    Returns (frames, n_mfcc) float32. Not as accurate as librosa, but pure-numpy."""
    win_len = int(rate * win_ms / 1000)
    hop_len = int(rate * hop_ms / 1000)
    pad = (n_fft - win_len) // 2
    if pad > 0:
        win_len = n_fft

    # Frame the signal
    n_frames = 1 + max(0, (len(signal) - win_len) // hop_len)
    frames = np.stack([signal[i * hop_len: i * hop_len + win_len]
                       for i in range(n_frames)], axis=0)
    # Hann window
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_len) / max(1, win_len - 1))
    frames = frames * hann
    # FFT magnitude
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1))

    # Mel filterbank
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    low = hz_to_mel(0)
    high = hz_to_mel(rate / 2)
    mels = np.linspace(low, high, n_mel + 2)
    hz_pts = mel_to_hz(mels)
    bin_pts = np.floor((n_fft + 1) * hz_pts / rate).astype(int)
    fb = np.zeros((n_mel, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mel):
        left, center, right = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        fb[i, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        fb[i, center:right] = (right - np.arange(center, right)) / max(1, right - center)
    mel_spec = (spec ** 2) @ fb.T
    log_mel = np.log(mel_spec + 1e-6)

    # DCT-II of log-mel → MFCC (truncate to n_mfcc)
    dct = np.cos(np.pi / n_mel * (np.arange(n_mel) + 0.5)[:, None] * np.arange(n_mfcc)[None, :])
    mfcc = log_mel @ dct
    return mfcc.astype(np.float32)


# -------------------- Tiny DS-CNN --------------------

class TinyDSCNN(nn.Module):
    def __init__(self, n_classes: int, ch: int = 16, n_mfcc: int = 10) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_mfcc)
        self.conv1 = nn.Conv1d(n_mfcc, ch, kernel_size=5, padding=2)
        # Depthwise + pointwise blocks
        self.dw1 = nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pw1 = nn.Conv1d(ch, ch * 2, kernel_size=1)
        self.dw2 = nn.Conv1d(ch * 2, ch * 2, kernel_size=3, padding=1, groups=ch * 2)
        self.pw2 = nn.Conv1d(ch * 2, ch * 2, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, frames, n_mfcc) → (B, n_mfcc, frames)
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.pw1(F.relu(self.dw1(x))))
        x = F.relu(self.pw2(F.relu(self.dw2(x))))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    data_path = ROOT.parent.parent / "datasets" / "kws_synthetic.npz"
    if not data_path.exists():
        raise SystemExit(f"missing {data_path}; run generate_synthetic.py first")
    blob = np.load(data_path, allow_pickle=True)
    X_raw, y = blob["X"], blob["y"]
    names = list(blob["names"])
    print(f"audio: {X_raw.shape}  classes={names}")

    # MFCC every sample
    mfccs = np.stack([
        compute_mfcc(x, cfg["sample_rate_hz"], cfg["window_ms"], cfg["hop_ms"], cfg["n_mfcc"])
        for x in X_raw
    ], axis=0)
    print(f"mfcc shape: {mfccs.shape}  (samples, frames, n_mfcc)")

    # Train / val split
    idx = np.arange(len(mfccs))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(mfccs[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(mfccs[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_dl = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg["batch_size"])

    model = TinyDSCNN(n_classes=cfg["n_classes"], n_mfcc=cfg["n_mfcc"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} params (~{n_params * 4 / 1024:.1f} KB FP32, "
          f"~{n_params / 1024:.1f} KB INT8)")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0; correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb)
                val_loss += F.cross_entropy(logits, yb, reduction="sum").item()
                correct += int((logits.argmax(dim=1) == yb).sum())
        val_loss /= len(X_val)
        val_acc = correct / len(X_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:>2}/{cfg['epochs']}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    out = ROOT / "results"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "names": names,
                "n_params": n_params, "mfcc_shape": list(mfccs.shape[1:])},
               out / "kws_model.pt")
    print(f"saved {out / 'kws_model.pt'}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs_axis = range(1, cfg["epochs"] + 1)
    axes[0].plot(epochs_axis, history["train_loss"], label="train")
    axes[0].plot(epochs_axis, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].legend()
    axes[0].set_title("loss"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs_axis, history["val_acc"], marker="o")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val accuracy")
    axes[1].set_title(f"val accuracy (final={history['val_acc'][-1]:.3f})")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=120)
    print(f"saved {out / 'training_curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
