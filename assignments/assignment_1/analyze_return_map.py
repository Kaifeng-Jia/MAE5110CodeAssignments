"""Sample the return map and estimate the rolling Floquet multiplier."""

import json
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
from analyze_roa import ASSIGNMENT_DIRECTORY, save_configuration
from models import rimless_wheel
from simulate_wheel import simulate_wheel


def create_configuration():
    """Define the return-map and perturbation experiments."""
    parameters = rimless_wheel.create_parameters()
    step_start_angle, _ = rimless_wheel.calculate_contact_angles(parameters)
    return {
        "parameters": parameters,
        "initial_state": [step_start_angle, 0.5],
        "simulation": {"number_of_steps": 30, "time_step": 0.01, "max_time": 30.0},
        "section": "Immediately after forward contact",
        "units": {"angle": "rad", "angular_velocity": "rad/s", "time": "s"},
        "prediction": "Post-impact speeds approach the rolling value near 1.971435 rad/s",
        "fixed_point_estimation": "Final post-impact speed; retain the last speed change",
        "floquet": {
            "velocity_perturbation": 0.01,
            "number_of_steps": 1,
            "method": "Centered difference across the fixed-point estimate",
            "prediction": "Local slope near 0.5",
        },
    }


def run_return_map(configuration):
    """Collect successive post-impact velocities along one trajectory."""
    times, states, reason = simulate_wheel(
        configuration["initial_state"],
        configuration["parameters"],
        **configuration["simulation"],
    )
    impact_indices = np.flatnonzero(np.diff(times) == 0) + 1
    velocities = np.concatenate(([states[1, 0]], states[1, impact_indices]))
    return {
        "times": times,
        "states": states,
        "post_impact_velocities": velocities,
        "fixed_point_estimate": velocities[-1],
        "last_velocity_change": abs(velocities[-1] - velocities[-2]),
        "stop_reason": reason,
    }


def estimate_floquet_multiplier(fixed_point, configuration):
    """Estimate the local slope using two perturbed one-step simulations."""
    perturbation = configuration["floquet"]["velocity_perturbation"]
    input_velocities = fixed_point + np.array([-perturbation, perturbation])
    output_velocities = []
    stop_reasons = []
    impact_counts = []
    simulation = {
        **configuration["simulation"],
        "number_of_steps": configuration["floquet"]["number_of_steps"],
    }
    for angular_velocity in input_velocities:
        times, states, reason = simulate_wheel(
            [configuration["initial_state"][0], angular_velocity],
            configuration["parameters"],
            **simulation,
        )
        output_velocities.append(states[1, -1])
        stop_reasons.append(reason)
        impact_counts.append(np.count_nonzero(np.diff(times) == 0))

    return {
        "perturbed_velocities": input_velocities,
        "next_velocities": np.array(output_velocities),
        "perturbation_stop_reasons": np.array(stop_reasons),
        "perturbation_impact_counts": np.array(impact_counts),
        "floquet_multiplier": (output_velocities[1] - output_velocities[0])
        / (2 * perturbation),
    }


def plot_return_map(run_directory, figure_path):
    """Plot adjacent velocity pairs, the identity line, and the fixed-point estimate."""
    configuration = json.loads((run_directory / "config.json").read_text())
    with np.load(run_directory / "results.npz", allow_pickle=False) as results:
        velocities = results["post_impact_velocities"]
        fixed_point = float(results["fixed_point_estimate"])

    limits = [velocities.min() - 0.1, velocities.max() + 0.15]
    figure, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    axis.plot(
        velocities[:-1], velocities[1:], "o-", color="tab:blue",
        markersize=5, label="Samples from one trajectory",
    )
    axis.plot(limits, limits, "--", color="0.4", label=r"Identity: $y=x$")
    axis.scatter(
        fixed_point, fixed_point, marker="o", s=70, color="red",
        zorder=3, label=fr"Estimated fixed point: $\omega^*\approx{fixed_point:.6f}$",
    )
    parameters = configuration["parameters"]
    axis.set(
        xlim=limits, ylim=limits, aspect="equal",
        xlabel=r"Post-impact velocity $\omega_n$ (rad/s)",
        ylabel=r"Next post-impact velocity $\omega_{n+1}$ (rad/s)",
        title=(
            f"Return-map samples: {len(velocities) - 1} steps\n"
            f"{parameters['spoke_count']} spokes, "
            f"slope = {np.rad2deg(parameters['slope_angle']):g} deg"
        ),
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def main():
    """Save the return-map samples, figure, and Floquet estimate."""
    configuration = create_configuration()
    run_id = datetime.now(UTC).strftime("return_map_%Y%m%dT%H%M%S%fZ")
    run_directory = ASSIGNMENT_DIRECTORY / "results" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    save_configuration(configuration, run_directory)
    results = run_return_map(configuration)
    results.update(estimate_floquet_multiplier(results["fixed_point_estimate"], configuration))
    np.savez_compressed(run_directory / "results.npz", **results)
    plot_return_map(run_directory, ASSIGNMENT_DIRECTORY / "figures" / f"{run_id}.png")
    print(
        f"Floquet multiplier: {results['floquet_multiplier']:.6f}; saved results to "
        f"{run_directory.relative_to(ASSIGNMENT_DIRECTORY)}"
    )


if __name__ == "__main__":
    main()
