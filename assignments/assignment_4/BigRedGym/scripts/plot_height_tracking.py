"""Build the quantitative PDF from the same live recording's measurements."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "deliverables"
with (OUT / "height_trace.csv").open() as stream:
    records = list(csv.DictReader(stream))
metadata = json.loads((OUT / "recording_metadata.json").read_text())
t = np.array([float(row["time_s"]) for row in records])
target = np.array([float(row["target_height_m"]) for row in records])
actual = np.array([float(row["actual_height_m"]) for row in records])
error = actual - target
assert len(records) == 2000
assert np.isfinite(np.concatenate([t, target, actual])).all()
assert np.all(np.diff(t) > 0)
assert metadata["velocity_commands_zero"]
assert not metadata["terminated_any"]

labels = ["LOW 1/2", "HIGH 1/2", "LOW 2/2", "HIGH 2/2"]
stages = []
for index, label in enumerate(labels):
    selected = (t > (index + 1) * 5 - 1) & (t <= (index + 1) * 5)
    stages.append(
        {
            "stage": label,
            "target_height_m": float(target[selected][-1]),
            "last_second_mean_height_m": float(actual[selected].mean()),
            "last_second_mae_m": float(np.abs(error[selected]).mean()),
            "last_second_std_m": float(actual[selected].std()),
        }
    )
metrics = {
    "duration_s": float(t[-1]),
    "samples": len(records),
    "mae_m": float(np.abs(error).mean()),
    "rmse_m": float(np.sqrt(np.mean(error**2))),
    "max_absolute_error_m": float(np.abs(error).max()),
    "termination_seen": metadata["terminated_any"],
    "stages": stages,
}
(OUT / "height_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig = plt.figure(figsize=(8.5, 11))
fig.text(0.10, 0.955, "MAE 5110 | Assignment 4", fontsize=18, weight="bold")
fig.text(
    0.10, 0.922, "Checkpoint: 550 iterations | 4096 training environments", fontsize=9
)
axes = fig.add_gridspec(
    2, 1, left=0.11, right=0.96, top=0.88, bottom=0.36, hspace=0.38
).subplots(sharex=True)
axes[0].plot(
    t,
    target,
    color="#d66b16",
    linestyle="--",
    linewidth=1.8,
    drawstyle="steps-post",
    label="Target height",
)
axes[0].plot(t, actual, color="#155a8a", linewidth=1.5, label="Measured height")
axes[0].set_ylabel("Base height (m)")
axes[0].set_ylim(0.12, 0.52)
axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)
axes[1].plot(t, error * 100, color="#155a8a", linewidth=1.2)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Actual - target (cm)")
axes[1].set_xlabel("Simulation time (s)")
for ax in axes:
    ax.set_xlim(0, 20)
    ax.set_xticks(np.arange(0, 21, 2.5))
    ax.grid(alpha=0.2)
    for boundary in (5, 10, 15):
        ax.axvline(boundary, color="0.55", linewidth=0.8, linestyle=":")
for index, label in enumerate(labels):
    axes[0].text(
        index * 5 + 2.5, 0.505, label, ha="center", va="top", fontsize=9, color="0.3"
    )
fig.text(
    0.10,
    0.305,
    f"Overall MAE: {metrics['mae_m'] * 100:.2f} cm     "
    f"RMSE: {metrics['rmse_m'] * 100:.2f} cm",
    fontsize=11,
    weight="bold",
)
fig.text(
    0.10, 0.275, "Statistics for the final 1 second of each 5-second stage", fontsize=10
)
table_ax = fig.add_axes([0.10, 0.08, 0.86, 0.17])
table_ax.axis("off")
cells = [
    [
        stage["stage"],
        f"{stage['target_height_m']:.2f}",
        f"{stage['last_second_mean_height_m']:.3f}",
        f"{stage['last_second_mae_m'] * 100:.2f}",
        f"{stage['last_second_std_m'] * 100:.2f}",
    ]
    for stage in stages
]
table = table_ax.table(
    cellText=cells,
    colLabels=["Stage", "Target (m)", "Measured (m)", "MAE (cm)", "SD (cm)"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for (row, column), cell in table.get_celld().items():
    cell.set_edgecolor("#d2d8df")
    if row == 0:
        cell.set_facecolor("#e8eef4")
        cell.set_text_props(weight="bold")
fig.savefig(OUT / "height_tracking.pdf")
fig.savefig(OUT / "height_tracking.png", dpi=150)
plt.close(fig)
print("METRICS", json.dumps(metrics))
print("PDF", OUT / "height_tracking.pdf")
