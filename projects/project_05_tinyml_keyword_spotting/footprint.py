"""Estimate the memory footprint of the trained KWS model on a microcontroller.

We do not actually quantize-to-TFLite here (that needs the tensorflow pip package).
We compute the *budgets* you would use to decide whether to flash to a board.

For real INT8 conversion, see Chapter 8 (ONNX quantization) and Chapter 12 doc
(TFLite Micro conversion pipeline).
"""
from __future__ import annotations

from pathlib import Path

import torch

from train_kws import TinyDSCNN


ROOT = Path(__file__).resolve().parent


def estimate_activation_peak(model: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """Rough estimate: track sizeof(activation) at each module via hooks,
    return the max tensor byte size seen."""
    peak = {"bytes": 0}

    def hook(_module, _inputs, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        b = t.numel() * t.element_size()
        if b > peak["bytes"]:
            peak["bytes"] = b

    handles = [m.register_forward_hook(hook)
               for m in model.modules()
               if not isinstance(m, (torch.nn.Sequential, type(model)))]
    try:
        with torch.no_grad():
            model(torch.zeros(input_shape))
    finally:
        for h in handles:
            h.remove()
    return peak["bytes"]


def main() -> int:
    ckpt_path = ROOT / "results" / "kws_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing {ckpt_path}; run train_kws.py first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt["config"]
    mfcc_shape = ckpt["mfcc_shape"]   # (frames, n_mfcc)
    model = TinyDSCNN(n_classes=cfg["n_classes"], n_mfcc=cfg["n_mfcc"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n_params = ckpt["n_params"]
    flash_fp32_kb = n_params * 4 / 1024
    flash_int8_kb = n_params / 1024

    # Estimate activation peak at FP32 (then divide by 4 for INT8)
    inp_shape = (1, mfcc_shape[0], mfcc_shape[1])
    peak_bytes_fp32 = estimate_activation_peak(model, inp_shape)
    peak_kb_fp32 = peak_bytes_fp32 / 1024
    peak_kb_int8 = peak_kb_fp32 / 4

    print(f"=== Estimated TinyML footprint for kws_model ===")
    print(f"params               : {n_params}")
    print(f"input shape (MFCC)   : {tuple(inp_shape)}  (1 second of audio)")
    print()
    print(f"FLASH (FP32 weights) : {flash_fp32_kb:7.1f} KB")
    print(f"FLASH (INT8 weights) : {flash_int8_kb:7.1f} KB  ← typical TinyML build")
    print()
    print(f"RAM activation peak (FP32): {peak_kb_fp32:7.1f} KB")
    print(f"RAM activation peak (INT8): {peak_kb_int8:7.1f} KB  ← typical TinyML build")
    print()
    print(f"=== Does it fit? ===")
    boards = [
        ("Arduino Nano 33 BLE Sense", 256, 1024),
        ("Raspberry Pi Pico",         264, 2048),
        ("ESP32-S3 (no PSRAM)",       512, 8192),
        ("STM32 H7",                 1024, 2048),
    ]
    int8_fits = lambda ram_kb, flash_kb: (peak_kb_int8 <= ram_kb * 0.5 and
                                          flash_int8_kb <= flash_kb * 0.5)
    print(f"{'board':<32s} {'RAM KB':>8s} {'Flash KB':>10s} {'fits INT8?':>12s}")
    for name, ram_kb, flash_kb in boards:
        ok = "YES" if int8_fits(ram_kb, flash_kb) else "TIGHT/NO"
        print(f"{name:<32s} {ram_kb:>8d} {flash_kb:>10d} {ok:>12s}")
    print("\n(rule of thumb: leave ≥50% of RAM/Flash headroom for the framework + user code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
