"""Estimate and plot rimless-wheel regions of attraction."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from models import rimless_wheel
from simulate_wheel import (
    EVENT_ANGLE_TOLERANCE,
    EVENT_TIME_TOLERANCE,
    STANDING_VELOCITY_TOLERANCE,
    simulate_wheel,
)

ASSIGNMENT_DIRECTORY = Path(__file__).resolve().parent


def create_configuration():
    """Define the initial-state grid and convergence criteria."""
    parameters = rimless_wheel.create_parameters()
    step_start_angle, step_end_angle = rimless_wheel.calculate_contact_angles(parameters)
    return {
        "scope": "RoA grid for the configured slope and spoke count",
        "prediction": "Standing and rolling basins, with rocking at low energy",
        "parameters": parameters,
        "units": {
            "angle": "rad",
            "angular_velocity": "rad/s",
            "time": "s",
            "gravity": "m/s^2",
            "spoke_length": "m",
            "hub_mass": "kg",
        },
        "grid": {
            "angle_min": step_start_angle,
            "angle_max": step_end_angle,
            "angle_count": 31,
            "angular_velocity_min": -3.0,
            "angular_velocity_max": 3.0,
            "angular_velocity_count": 31,
        },
        "simulation": {
            "number_of_steps": 30,
            "time_step": 0.01,
            "max_time": 30.0,
            "standing_velocity_tolerance": STANDING_VELOCITY_TOLERANCE,
        },
        "event_angle_tolerance": EVENT_ANGLE_TOLERANCE,
        "event_time_tolerance": EVENT_TIME_TOLERANCE,
        "outcome_codes": {"0": "unresolved", "1": "standing", "2": "rolling"},
        "plot": "Separate standing and rolling basin maps; other outcomes left blank",
        "classification": {
            "minimum_impacts": 10,
            "recent_changes": 5,
            "velocity_tolerance": 1e-6,
            "standing": "Contact boundary, near-zero speed, below both tipping barriers",
        },
    }


def classify_trajectory(times, states, reason, criterion):
    """Return the attractor code, impact count, and recent speed change."""
    impact_indices = np.flatnonzero(np.diff(times) == 0) + 1
    post_impact_velocities = states[1, impact_indices]
    velocity_changes = np.abs(np.diff(post_impact_velocities))
    recent_count = criterion["recent_changes"]
    maximum_change = (
        float(np.max(velocity_changes[-recent_count:]))
        if len(velocity_changes) >= recent_count
        else np.nan
    )
    if reason == "standing":
        outcome = 1
    elif (
        reason == "step_limit"
        and len(impact_indices) >= criterion["minimum_impacts"]
        and np.all(post_impact_velocities[-recent_count - 1 :] > 0)
        and maximum_change < criterion["velocity_tolerance"]
    ):
        outcome = 2
    else:
        outcome = 0
    return outcome, len(impact_indices), maximum_change


def run_grid(configuration):
    """Simulate each grid point and retain its classification measurements."""
    grid = configuration["grid"]
    # Remove roundoff at zero in the initial-angle grid.
    angles = np.linspace(grid["angle_min"], grid["angle_max"], grid["angle_count"]).round(14)
    angular_velocities = np.linspace(
        grid["angular_velocity_min"],
        grid["angular_velocity_max"],
        grid["angular_velocity_count"],
    )
    shape = (len(angular_velocities), len(angles))
    results = {
        "angles": angles,
        "angular_velocities": angular_velocities,
        "outcomes": np.zeros(shape, dtype=np.int8),
        "stop_reasons": np.empty(shape, dtype="U32"),
        "termination_times": np.zeros(shape),
        "impact_counts": np.zeros(shape, dtype=int),
        "max_recent_velocity_changes": np.full(shape, np.nan),
        "final_states": np.zeros((*shape, 2)),
    }
    for row, angular_velocity in enumerate(angular_velocities):
        for column, angle in enumerate(angles):
            times, states, reason = simulate_wheel(
                [angle, angular_velocity],
                configuration["parameters"],
                **configuration["simulation"],
            )
            outcome, impact_count, maximum_change = classify_trajectory(
                times, states, reason, configuration["classification"]
            )
            results["outcomes"][row, column] = outcome
            results["stop_reasons"][row, column] = reason
            results["termination_times"][row, column] = times[-1]
            results["impact_counts"][row, column] = impact_count
            results["max_recent_velocity_changes"][row, column] = maximum_change
            results["final_states"][row, column] = states[:, -1]
    return results


def save_configuration(configuration, run_directory):
    """Save the configuration, environment versions, and actual source snapshot."""
    repository = ASSIGNMENT_DIRECTORY.parents[1]
    source_files = list(ASSIGNMENT_DIRECTORY.glob("*.py"))
    for package in ("models", "integrators"):
        source_files.extend((ASSIGNMENT_DIRECTORY / package).glob("*.py"))
    source_files.extend(repository / name for name in ("pyproject.toml", "uv.lock"))
    snapshot_path = run_directory / "source.zip"
    with ZipFile(snapshot_path, "w", compression=ZIP_DEFLATED) as snapshot:
        for source_path in sorted(source_files):
            if source_path.exists():
                snapshot.write(source_path, source_path.relative_to(repository))

    saved_configuration = {
        **configuration,
        "run_id": run_directory.name,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "source_snapshot": "source.zip",
        "source_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }
    (run_directory / "config.json").write_text(
        json.dumps(saved_configuration, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def plot_roa(run_directory, figure_directory):
    """Save one figure per attractor from the saved grid results."""
    configuration = json.loads(
        (run_directory / "config.json").read_text(encoding="utf-8")
    )
    parameters = configuration["parameters"]
    attractors = [
        (1, "standing", "Standing equilibrium", "#4477AA"),
        (2, "rolling", "Rolling limit cycle", "#EEAA55"),
    ]
    figure_directory.mkdir(parents=True, exist_ok=True)
    with np.load(run_directory / "results.npz", allow_pickle=False) as results:
        angles = np.rad2deg(results["angles"])
        angular_velocities = results["angular_velocities"]
        for code, name, label, color in attractors:
            basin_count = np.count_nonzero(results["outcomes"] == code)
            figure, axis = plt.subplots(figsize=(8, 6), layout="constrained")
            axis.pcolormesh(
                angles,
                angular_velocities,
                np.ma.masked_where(results["outcomes"] != code, results["outcomes"]),
                cmap=ListedColormap([color]),
                shading="nearest",
            )
            axis.set(
                xlim=(angles[0], angles[-1]),
                ylim=(angular_velocities[0], angular_velocities[-1]),
                xlabel=r"Initial stance angle $\theta_0$ (deg)",
                ylabel=r"Initial angular velocity $\dot{\theta}_0$ (rad/s)",
                title=(
                    f"{label}: estimated region of attraction\n"
                    f"{parameters['spoke_count']} spokes, "
                    f"slope = {np.rad2deg(parameters['slope_angle']):g} deg"
                ),
            )
            axis.legend(
                handles=[Patch(color=color, label=f"{label} basin: {basin_count} grid points")],
                loc="upper left",
            )
            figure.savefig(figure_directory / f"{run_directory.name}_{name}.png", dpi=160)
            plt.close(figure)


def main():
    """Save one grid experiment and a figure for each attractor."""
    configuration = create_configuration()
    run_id = datetime.now(UTC).strftime("roa_%Y%m%dT%H%M%S%fZ")
    run_directory = ASSIGNMENT_DIRECTORY / "results" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    save_configuration(configuration, run_directory)
    results = run_grid(configuration)
    np.savez_compressed(run_directory / "results.npz", **results)
    plot_roa(run_directory, ASSIGNMENT_DIRECTORY / "figures")
    print(
        f"Saved {results['outcomes'].size} initial states to "
        f"{run_directory.relative_to(ASSIGNMENT_DIRECTORY)}"
    )


if __name__ == "__main__":
    main()
