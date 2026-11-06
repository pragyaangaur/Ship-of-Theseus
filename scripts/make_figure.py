"""Draws figures/ship.png and figures/ship.pdf from results/main/table.json.

Left panel: self-report identity against the fraction of the ship taken from DeepSeek. Right panel: WikiText perplexity on the same x axis. The two measures have different scales, so they get separate panels.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NP = 31
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#2b2b2a", "#6b6a63", "#e4e3dd"

table = {r["id"]: r for r in json.loads((ROOT / "results/main/table.json").read_text())}
A, B = table["ctrl/A"], table["ctrl/B"]


def plank_series(order, key):
    xs = [0.0] + [k / NP for k in range(1, NP)] + [1.0]
    ys = [0.0 if key == "iota_A" else A[key]]
    ys += [table[f"arm1/{order}/k{k:02d}"][key] for k in range(1, NP)]
    ys += [1.0 if key == "iota_A" else B[key]]
    return xs, ys


def blend_series(key):
    xs = [0.0] + [j / 10 for j in range(1, 10)] + [1.0]
    ys = [0.0 if key == "iota_A" else A[key]]
    ys += [table[f"arm3/a{j / 10}"][key] for j in range(1, 10)]
    ys += [1.0 if key == "iota_A" else B[key]]
    return xs, ys


SERIES = [
    ("Planks from the input end", lambda k: plank_series("forward", k), BLUE, "o"),
    ("Planks from the output end", lambda k: plank_series("backward", k), ORANGE, "s"),
    ("Every weight blended", blend_series, AQUA, "^"),
]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)

for ax, key, ylabel in ((axes[0], "iota_A", "Says it is DeepSeek (0 = Qwen, 1 = DeepSeek)"),
                        (axes[1], "ppl", "WikiText perplexity (log scale)")):
    for label, fn, color, marker in SERIES:
        xs, ys = fn(key)
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=4.5, mec="white", mew=0.8, label=label, zorder=3)
    ax.set_xlabel("Fraction of the ship taken from DeepSeek")
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(-0.02, 1.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
axes[0].set_title("What the ship says it is", loc="left", fontsize=10, color=INK)
axes[1].set_yscale("log")
axes[1].axhline(2 * max(A["ppl"], B["ppl"]), color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
axes[1].text(0.47, 2 * max(A["ppl"], B["ppl"]) * 0.72, "seaworthy limit", color=MUTED, fontsize=8)
axes[1].set_title("Whether the ship still floats", loc="left", fontsize=10, color=INK)
axes[0].legend(frameon=False, loc="lower right", fontsize=8)

out = ROOT / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "ship.png", dpi=200)
fig.savefig(out / "ship.pdf")
print("wrote", out / "ship.png")
