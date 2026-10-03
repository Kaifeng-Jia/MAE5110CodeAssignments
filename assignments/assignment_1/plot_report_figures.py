"""Arrange saved experiment results into report figures."""

import json
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
from analyze_return_map import plot_return_map
from analyze_roa import ASSIGNMENT_DIRECTORY, save_configuration
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

BASIN_NAMES = {1: "Standing", 2: "Rolling"}
BASIN_COLORS = {1: "#4477AA", 2: "#EEAA55"}


def create_configuration():
    """Identify the saved experiments and report layouts."""
    return {
        "scope": "Report figures drawn from saved results",
        "sources": {
            "baseline": "roa_20260908T040439975194Z",
            "return_map": "return_map_20260908T043443824723Z",
            "slopes": "slope_sweep_20260908T044528669557Z",
            "spokes": "spoke_sweep_20260908T050144131805Z",
        },
        "layouts": {"baseline": [1, 2], "slopes": [2, 3], "spokes": [2, 4]},
        "axes": "Absolute stance angle; common limits within each comparison",
        "colors": "Blue standing; orange rolling; gray outside sampled angle interval",
        "blank_regions": "Initial states not in the basin shown, including unresolved states",
    }


def load_basin(run_directory):
    """Read the state grid and its classifications."""
    with np.load(run_directory / "results.npz", allow_pickle=False) as results:
        return {
            "angles": np.rad2deg(results["angles"]),
            "velocities": results["angular_velocities"],
            "outcomes": results["outcomes"],
        }


def calculate_cell_edges(points):
    """Bound grid cells by midpoints and the sampled interval endpoints."""
    return np.r_[points[0], (points[:-1] + points[1:]) / 2, points[-1]]


def plot_basin_panels(panels, shape, title, figure_path):
    """Draw comparable basin panels with shared axes and a common legend."""
    rows, columns = shape
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.6 * columns, 3.1 * rows + 1),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    limits = [
        min(basin["angles"][0] for basin, _, _ in panels),
        max(basin["angles"][-1] for basin, _, _ in panels),
    ]
    for axis, (basin, outcome, label) in zip(axes.flat, panels):
        angles, velocities = basin["angles"], basin["velocities"]
        axis.axvspan(limits[0], angles[0], color="0.93", zorder=0)
        axis.axvspan(angles[-1], limits[1], color="0.93", zorder=0)
        axis.pcolormesh(
            calculate_cell_edges(angles), calculate_cell_edges(velocities),
            np.ma.masked_where(basin["outcomes"] != outcome, basin["outcomes"]),
            cmap=ListedColormap([BASIN_COLORS[outcome]]), shading="flat",
        )
        count = np.count_nonzero(basin["outcomes"] == outcome)
        axis.set(
            xlim=limits, ylim=(velocities[0], velocities[-1]),
            title=f"{label}{BASIN_NAMES[outcome]}: {count}/{basin['outcomes'].size}",
        )
        axis.locator_params(axis="x", nbins=5)

    codes = dict.fromkeys(outcome for _, outcome, _ in panels)
    handles = [Patch(color=BASIN_COLORS[code], label=BASIN_NAMES[code]) for code in codes]
    handles.extend([
        Patch(facecolor="white", edgecolor="0.6", label="Not in the basin shown"),
        Patch(color="0.93", label="Outside sampled angle interval"),
    ])
    for axis in axes.flat[len(panels):]:
        axis.set_axis_off()
    figure.suptitle(title, fontsize=15)
    figure.supxlabel(r"Initial stance angle $\theta_0$ (deg)")
    figure.supylabel(r"Initial angular velocity $\dot{\theta}_0$ (rad/s)")
    if len(panels) < axes.size:
        axes.flat[-1].legend(handles=handles, loc="center", frameon=False)
    else:
        figure.set_layout_engine(None)
        height = figure.get_figheight()
        figure.subplots_adjust(
            left=0.09, right=0.99, top=1 - 0.65 / height,
            bottom=1.2 / height, hspace=0.3, wspace=0.08,
        )
        figure.supxlabel(r"Initial stance angle $\theta_0$ (deg)", y=0.65 / height)
        figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def plot_floquet_comparison(slope_summary, spoke_summary, figure_path):
    """Compare the saved Floquet estimates against slope and spoke count."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    comparisons = [
        (slope_summary, "slope_degrees", r"Slope $\gamma$ (deg)", "8 spokes"),
        (spoke_summary, "spoke_count", "Number of spokes", "Slope = 15 deg"),
    ]
    for axis, (summary, parameter, label, title) in zip(axes, comparisons):
        values = [case[parameter] for case in summary]
        axis.plot(values, [case["floquet_multiplier"] for case in summary], "o-", color="#4477AA")
        axis.set(xlabel=label, title=title, xticks=values, ylim=(0, 1))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"Estimated Floquet multiplier $\lambda$")
    figure.suptitle("Local convergence per impact", fontsize=15)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def main():
    """Save report figures and the source record without simulating."""
    configuration = create_configuration()
    run_id = datetime.now(UTC).strftime("report_figures_%Y%m%dT%H%M%S%fZ")
    run_directory = ASSIGNMENT_DIRECTORY / "results" / run_id
    figure_directory = ASSIGNMENT_DIRECTORY / "figures" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    figure_directory.mkdir(parents=True, exist_ok=False)
    save_configuration(configuration, run_directory)
    sources = {
        name: ASSIGNMENT_DIRECTORY / "results" / run
        for name, run in configuration["sources"].items()
    }
    baseline = load_basin(sources["baseline"])
    plot_basin_panels(
        [(baseline, code, "") for code in (1, 2)], configuration["layouts"]["baseline"],
        "Baseline regions of attraction: 8 spokes, slope = 15 deg",
        figure_directory / "baseline_roa.png",
    )
    plot_return_map(sources["return_map"], figure_directory / "return_map.png")

    slope_summary = json.loads((sources["slopes"] / "summary.json").read_text())
    slope_basins = [load_basin(sources["slopes"] / case["case_directory"]) for case in slope_summary]
    plot_basin_panels(
        [(basin, code, f"{case['slope_degrees']:g} deg | ")
         for code in (1, 2) for basin, case in zip(slope_basins, slope_summary)],
        configuration["layouts"]["slopes"], "Slope comparison: 8 spokes",
        figure_directory / "slope_roa.png",
    )
    spoke_summary = json.loads((sources["spokes"] / "summary.json").read_text())
    spoke_basins = [load_basin(sources["spokes"] / case["case_directory"]) for case in spoke_summary]
    for code in (1, 2):
        plot_basin_panels(
            [(basin, code, f"N={case['spoke_count']} | ") for basin, case in zip(spoke_basins, spoke_summary)],
            configuration["layouts"]["spokes"], "Spoke-count comparison: slope = 15 deg",
            figure_directory / f"spokes_{BASIN_NAMES[code].lower()}_roa.png",
        )
    plot_floquet_comparison(slope_summary, spoke_summary, figure_directory / "floquet_comparison.png")
    print(f"Saved six report figures to {figure_directory.relative_to(ASSIGNMENT_DIRECTORY)}")


if __name__ == "__main__":
    main()
