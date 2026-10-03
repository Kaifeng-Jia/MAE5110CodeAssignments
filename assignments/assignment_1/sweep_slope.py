"""Compare regions of attraction and local convergence at three slopes."""

import json
from datetime import UTC, datetime

import analyze_return_map
import analyze_roa
import numpy as np
from analyze_roa import ASSIGNMENT_DIRECTORY
from models import rimless_wheel


def create_configuration(slope_degrees):
    """Configure the grid and rolling experiment for one slope."""
    configuration = analyze_roa.create_configuration()
    parameters = configuration["parameters"]
    parameters["slope_angle"] = np.deg2rad(slope_degrees)
    step_start_angle, step_end_angle = rimless_wheel.calculate_contact_angles(parameters)
    configuration["scope"] = "Slope comparison with eight spokes"
    configuration["prediction"] = (
        "Standing basin shrinks as slope increases and disappears at 25 degrees"
    )
    configuration["grid"].update(
        angle_min=step_start_angle, angle_max=step_end_angle
    )

    return_map = analyze_return_map.create_configuration()
    return_map["parameters"] = parameters
    return_map["initial_state"] = [step_start_angle, 2.0]
    return_map["prediction"] = "Rolling speed increases with slope; local slope stays near 0.5"
    configuration["return_map"] = return_map
    return configuration


def run_case(configuration):
    """Collect the grid, rolling trajectory, and two-sided Floquet estimate."""
    results = analyze_roa.run_grid(configuration)
    return_map = configuration["return_map"]
    rolling_results = analyze_return_map.run_return_map(return_map)
    rolling_results.update(
        analyze_return_map.estimate_floquet_multiplier(
            rolling_results["fixed_point_estimate"], return_map
        )
    )
    results.update(rolling_results)
    return results


def summarize_case(configuration, run_directory, results):
    """Record attractor counts and the rolling convergence measurements."""
    return {
        "slope_degrees": float(np.rad2deg(configuration["parameters"]["slope_angle"])),
        "spoke_count": configuration["parameters"]["spoke_count"],
        "case_directory": run_directory.name,
        "standing_count": int(np.count_nonzero(results["outcomes"] == 1)),
        "rolling_count": int(np.count_nonzero(results["outcomes"] == 2)),
        "unresolved_count": int(np.count_nonzero(results["outcomes"] == 0)),
        "fixed_point_estimate": float(results["fixed_point_estimate"]),
        "last_velocity_change": float(results["last_velocity_change"]),
        "floquet_multiplier": float(results["floquet_multiplier"]),
        "rolling_impact_count": len(results["post_impact_velocities"]) - 1,
        "rolling_duration": float(results["times"][-1]),
        "grid_max_impact_count": int(np.max(results["impact_counts"])),
        "grid_max_duration": float(np.max(results["termination_times"])),
    }


def run_sweep(sweep_name, configurations):
    """Save the configured cases, basin figures, and comparison table."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    sweep_id = f"{sweep_name}_{timestamp}"
    sweep_directory = ASSIGNMENT_DIRECTORY / "results" / sweep_id
    figure_directory = ASSIGNMENT_DIRECTORY / "figures" / sweep_id
    summaries = []
    for case_name, configuration in configurations.items():
        run_directory = sweep_directory / case_name
        run_directory.mkdir(parents=True, exist_ok=False)
        analyze_roa.save_configuration(configuration, run_directory)
        results = run_case(configuration)
        np.savez_compressed(run_directory / "results.npz", **results)
        analyze_roa.plot_roa(run_directory, figure_directory)
        summaries.append(summarize_case(configuration, run_directory, results))

    (sweep_directory / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {len(summaries)} cases to "
        f"{sweep_directory.relative_to(ASSIGNMENT_DIRECTORY)}"
    )


def main():
    """Compare three slopes with eight spokes."""
    configurations = {
        f"slope_{slope_degrees:02d}deg": create_configuration(slope_degrees)
        for slope_degrees in (5, 15, 25)
    }
    run_sweep("slope_sweep", configurations)


if __name__ == "__main__":
    main()
