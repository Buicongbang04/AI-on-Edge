"""Generate the two figures for Chapter 0.

Run from the repo root:

    conda run -n aicourse python figures/_generate_chapter_00_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Figure 1: Cloud AI vs Edge AI inference path
# ---------------------------------------------------------------------------

def make_cloud_vs_edge() -> Path:
    fig, (ax_cloud, ax_edge) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Cloud AI vs Edge AI: where inference happens",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    def draw_box(ax, x, y, w, h, text, *, fc, ec="#222", fontweight="normal"):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, facecolor=fc, edgecolor=ec,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=10.5, fontweight=fontweight,
        )

    def draw_arrow(ax, x0, y0, x1, y1, *, label=None, color="#333", style="->",
                   lw=1.6, label_dy=0.12):
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle=style, mutation_scale=14,
            linewidth=lw, color=color,
        )
        ax.add_patch(arrow)
        if label:
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2 + label_dy, label,
                ha="center", va="bottom",
                fontsize=9, color=color, style="italic",
            )

    # ---- Cloud panel
    ax_cloud.set_title("Cloud AI", fontsize=12, fontweight="bold")
    ax_cloud.set_xlim(0, 10)
    ax_cloud.set_ylim(0, 10)
    ax_cloud.axis("off")

    draw_box(ax_cloud, 0.4, 4.0, 1.8, 1.4, "Device\n(camera /\nsensor)", fc="#E3F2FD")
    draw_box(ax_cloud, 4.0, 4.0, 2.0, 1.4, "Cloud server\n(GPU / TPU)\nmodel + inference", fc="#FFE082")
    draw_box(ax_cloud, 7.8, 4.0, 1.8, 1.4, "Device\n(action)", fc="#E3F2FD")

    draw_arrow(ax_cloud, 2.2, 4.7, 4.0, 4.7, label="upload raw data")
    draw_arrow(ax_cloud, 6.0, 4.7, 7.8, 4.7, label="return prediction")

    # network annotation (placed above the boxes to avoid overlap)
    ax_cloud.text(5.0, 8.4, "network round-trip", ha="center",
                  fontsize=10, fontweight="bold", color="#B71C1C")
    ax_cloud.text(5.0, 7.5, "~100-500 ms (cross-network)\n~30-60 ms (same region)",
                  ha="center", fontsize=9, color="#B71C1C")

    # constraints box
    constraints_cloud = [
        "+ Almost unlimited compute & RAM",
        "+ Centralized training and updates",
        "- High latency (network-bound)",
        "- High bandwidth: ships raw data",
        "- Privacy: data leaves the device",
        "- Useless when offline",
    ]
    ax_cloud.text(
        5.0, 2.2, "\n".join(constraints_cloud),
        ha="center", va="top", fontsize=9.5,
        family="monospace",
    )

    # ---- Edge panel
    ax_edge.set_title("Edge AI", fontsize=12, fontweight="bold")
    ax_edge.set_xlim(0, 10)
    ax_edge.set_ylim(0, 10)
    ax_edge.axis("off")

    draw_box(ax_edge, 0.6, 4.0, 1.8, 1.4, "Sensor\n(camera /\nmic / IMU)", fc="#E3F2FD")
    draw_box(ax_edge, 3.4, 4.0, 3.2, 1.4,
             "On-device inference\n(Jetson / RPi / NUC / MCU)\nmodel on-device",
             fc="#C8E6C9")
    draw_box(ax_edge, 7.6, 4.0, 1.8, 1.4, "Action\n(label /\nalert /\nactuator)", fc="#E3F2FD")

    draw_arrow(ax_edge, 2.4, 4.7, 3.4, 4.7, label="capture")
    draw_arrow(ax_edge, 6.6, 4.7, 7.6, 4.7, label="decide")

    ax_edge.text(5.0, 8.4, "local round-trip", ha="center",
                 fontsize=10, fontweight="bold", color="#1B5E20")
    ax_edge.text(5.0, 7.5, "~5-50 ms (on-device only)",
                 ha="center", fontsize=9, color="#1B5E20")

    constraints_edge = [
        "+ Low latency (no network)",
        "+ Low bandwidth: ships decisions only",
        "+ Privacy: data stays on the device",
        "+ Works offline",
        "- Limited RAM / compute / power",
        "- Needs export + optimization (Ch 5-8)",
    ]
    ax_edge.text(
        5.0, 2.2, "\n".join(constraints_edge),
        ha="center", va="top", fontsize=9.5,
        family="monospace",
    )

    # footer
    fig.text(
        0.5, 0.02,
        "Latency numbers are typical ranges from industry surveys (IBM, IJRMEET 2025, NVIDIA). "
        "Use them as orders of magnitude, not exact targets.",
        ha="center", fontsize=8.5, color="#555", style="italic",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path = FIG_DIR / "cloud_vs_edge.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Physical AI closed loop
# ---------------------------------------------------------------------------

def make_physical_ai_loop() -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        "Physical AI loop: perception → state → decision → action → feedback",
        fontsize=14, fontweight="bold", y=0.97,
    )
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def draw_box(x, y, w, h, text, *, fc, ec="#222", fontweight="normal"):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.5, facecolor=fc, edgecolor=ec,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=10.5, fontweight=fontweight,
        )

    def draw_arrow(x0, y0, x1, y1, *, color="#333", lw=1.8, style="->"):
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle=style, mutation_scale=18,
            linewidth=lw, color=color,
        )
        ax.add_patch(arrow)

    # Boxes along the top row (forward path)
    y_top = 5.2
    h = 1.4
    boxes = [
        ("Sensors\n(camera, LiDAR,\nmic, IMU)", 0.4, 2.0, "#E3F2FD"),
        ("Perception\n(detect / classify /\nsegment / pose)", 2.7, 2.2, "#C8E6C9"),
        ("State\n(world + self\nrepresentation)", 5.2, 2.0, "#FFE082"),
        ("Decision\n(rule-based or\nlearned policy / VLA)", 7.5, 2.2, "#FFCCBC"),
        ("Controller\n(joint cmd / PWM /\nvelocity)", 10.0, 2.2, "#D1C4E9"),
        ("Actuator\n(motor / arm /\ngripper / wheels)", 12.4, 2.0, "#F8BBD0"),
    ]
    centers = []
    for text, x, w, color in boxes:
        draw_box(x, y_top, w, h, text, fc=color)
        centers.append((x + w / 2, y_top + h / 2, x, x + w))

    # Forward arrows
    for i in range(len(centers) - 1):
        _, _, _, x_right = centers[i]
        _, _, x_left, _ = centers[i + 1]
        draw_arrow(x_right + 0.05, y_top + h / 2, x_left - 0.05, y_top + h / 2)

    # Safety layer overlay between Decision and Controller
    safety_x = 9.4
    safety_y = y_top - 1.4
    draw_box(safety_x - 0.85, safety_y, 1.7, 0.9,
             "Safety gate\n(limits / fallback)",
             fc="#FFCDD2", ec="#B71C1C", fontweight="bold")
    # Arrow from Decision → Safety, Safety → Controller
    draw_arrow(8.6, y_top + h / 2 - 0.6, safety_x - 0.05, safety_y + 0.9 / 2 + 0.05,
               color="#B71C1C", lw=1.6)
    draw_arrow(safety_x + 0.05, safety_y + 0.9 / 2 + 0.05,
               10.0 + 0.2, y_top + h / 2 - 0.6,
               color="#B71C1C", lw=1.6)
    ax.text(safety_x, safety_y - 0.5,
            "every action command\npasses through the safety layer",
            ha="center", va="top", fontsize=8.5, color="#B71C1C", style="italic")

    # Environment node
    env_y = 1.4
    draw_box(5.5, env_y, 3.0, 1.0,
             "Environment\n(the real world)", fc="#B3E5FC", fontweight="bold")

    # Arrow Actuator → Environment
    draw_arrow(13.4, y_top, 8.5 + 0.05, env_y + 1.0,
               color="#1565C0", lw=1.8)
    ax.text(11.5, 3.6, "action", color="#1565C0", fontsize=9, style="italic")

    # Feedback arrow Environment → Sensors
    draw_arrow(5.5, env_y + 0.5, 1.4, y_top - 0.05,
               color="#2E7D32", lw=1.8)
    ax.text(2.6, 3.6, "feedback\n(new observations)",
            color="#2E7D32", fontsize=9, style="italic")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="#C8E6C9", edgecolor="#222", label="Perception"),
        mpatches.Patch(facecolor="#FFE082", edgecolor="#222", label="State"),
        mpatches.Patch(facecolor="#FFCCBC", edgecolor="#222", label="Decision"),
        mpatches.Patch(facecolor="#D1C4E9", edgecolor="#222", label="Controller"),
        mpatches.Patch(facecolor="#FFCDD2", edgecolor="#B71C1C", label="Safety gate (mandatory)"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right",
        fontsize=9, frameon=True, ncol=1,
    )

    fig.text(
        0.5, 0.02,
        "Reference: NVIDIA Physical AI definition — perception, understanding, and complex action in the real world.",
        ha="center", fontsize=8.5, color="#555", style="italic",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_path = FIG_DIR / "physical_ai_loop.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p1 = make_cloud_vs_edge()
    p2 = make_physical_ai_loop()
    print(f"wrote: {p1}")
    print(f"wrote: {p2}")
