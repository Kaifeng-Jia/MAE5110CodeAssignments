import json
import shutil
from pathlib import Path

import numpy as np
from controllers import (
    compute_maximum_step_policy,
    compute_minimum_step_policy,
    is_in_roa,
    load_roa,
)
from models import inverted_pendulum_walker as model


def simulate_return(
    initial_angular_velocity,
    angle_of_attack,
    params,
    angles,
    angular_velocities,
    converged_grid,
    timestep=1e-4,
    duration=8.0,
):
    """Return the section velocity, RoA outcome, and phase trajectory."""
    state = np.array([0.0, initial_angular_velocity])
    states = [state]
    impact_indices = []
    next_velocity = None
    simulation_params = params.copy()
    simulation_params["angle_of_attack"] = angle_of_attack
    simulation_params["ankle_torque"] = 0.0

    reached_roa = is_in_roa(state, angles, angular_velocities, converged_grid)
    for step in range(round(duration / timestep)):
        if reached_roa:
            break
        next_state = state + timestep * model.dynamics(
            step * timestep, state, simulation_params
        )
        crossed_section = state[0] < 0 <= next_state[0] and next_state[1] > 0
        if crossed_section:
            # Interpolate to theta = 0.
            fraction = -state[0] / (next_state[0] - state[0])
            next_state = state + fraction * (next_state - state)
        states.append(next_state)

        reached_roa = is_in_roa(next_state, angles, angular_velocities, converged_grid)
        if reached_roa:
            break
        if crossed_section:
            next_velocity = next_state[1]
            break

        if model.event_guard(state, next_state, simulation_params):
            next_state = model.event_dynamics(next_state, simulation_params)
            states.append(next_state)
            impact_indices.append(len(states) - 1)
            reached_roa = is_in_roa(
                next_state, angles, angular_velocities, converged_grid
            )

        if (
            reached_roa
            or abs(next_state[0] - simulation_params["incline"]) >= np.pi / 2
        ):
            break
        state = next_state

    trajectory = {
        "states": np.array(states),
        "impact_indices": np.array(impact_indices, dtype=int),
    }
    return next_velocity, bool(reached_roa), trajectory


def build_transition_table(
    initial_velocities,
    angle_of_attack_values,
    params,
    angles,
    angular_velocities,
    converged_grid,
    timestep=1e-4,
    duration=8.0,
):
    """Tabulate each initial velocity and landing angle."""
    shape = (len(initial_velocities), len(angle_of_attack_values))
    next_velocities = np.full(shape, np.nan)
    reached_roa = np.zeros(shape, dtype=bool)
    initially_in_roa = np.array(
        [
            is_in_roa([0.0, velocity], angles, angular_velocities, converged_grid)
            for velocity in initial_velocities
        ]
    )

    for velocity_index, initial_velocity in enumerate(initial_velocities):
        for angle_index, angle_of_attack in enumerate(angle_of_attack_values):
            next_velocity, reached, _ = simulate_return(
                initial_velocity,
                angle_of_attack,
                params,
                angles,
                angular_velocities,
                converged_grid,
                timestep,
                duration,
            )
            reached_roa[velocity_index, angle_index] = reached
            if next_velocity is not None:
                next_velocities[velocity_index, angle_index] = next_velocity

    return next_velocities, reached_roa, initially_in_roa


def print_transition_table(
    initial_velocities,
    angle_of_attack_values,
    next_velocities,
    reached_roa,
    initially_in_roa,
):
    """Print return velocities and RoA outcomes."""
    print("Next angular velocity (rad/s)")
    header = ["omega0 (rad/s)"] + [
        f"alpha={np.rad2deg(alpha):.2f} deg" for alpha in angle_of_attack_values
    ]
    print(" | ".join(f"{label:>16}" for label in header))
    for velocity_index, initial_velocity in enumerate(initial_velocities):
        row = [f"{initial_velocity:.4f}"]
        for angle_index in range(len(angle_of_attack_values)):
            velocity = next_velocities[velocity_index, angle_index]
            if initially_in_roa[velocity_index]:
                value = "Initial RoA"
            elif reached_roa[velocity_index, angle_index]:
                value = "Reached RoA"
            elif np.isnan(velocity):
                value = "No return"
            else:
                value = f"{velocity:.4f}"
            row.append(value)
        print(" | ".join(f"{value:>16}" for value in row))


def print_step_policy(
    initial_velocities, step_counts, selected_angles, policy_name="Minimum"
):
    """Print the estimated step count and chosen angle for each state."""
    print(f"\n{policy_name}-step policy estimated from the grid")
    print(f"{'omega0 (rad/s)':>16} | {'steps':>12} | {'alpha (deg)':>16}")
    for velocity, steps, angle in zip(initial_velocities, step_counts, selected_angles):
        if steps == 0:
            action = "Balance"
        elif np.isfinite(steps):
            action = f"{np.rad2deg(angle):.2f}"
        else:
            action = "No policy"
        print(f"{velocity:>16.4f} | {steps:>12.0f} | {action:>16}")


def save_transition_table(
    initial_velocities,
    angle_of_attack_values,
    next_velocities,
    reached_roa,
    initially_in_roa,
    minimum_steps,
    selected_angles,
    maximum_steps,
    maximum_selected_angles,
    configuration,
):
    """Save transitions, policy estimates, settings, and source files."""
    folder = Path(__file__).resolve().parent
    output = folder / "output" / "transition_table"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "data.npz",
        initial_velocities=initial_velocities,
        angle_of_attack_values=angle_of_attack_values,
        next_velocities=next_velocities,
        reached_roa=reached_roa,
        initially_in_roa=initially_in_roa,
        minimum_steps=minimum_steps,
        selected_angles=selected_angles,
        maximum_steps=maximum_steps,
        maximum_selected_angles=maximum_selected_angles,
    )
    (output / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    for relative_path in (
        "poincare_map.py",
        "controllers.py",
        "models/__init__.py",
        "models/inverted_pendulum_walker.py",
    ):
        target = output / "source" / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / relative_path, target)
    return output


if __name__ == "__main__":
    params = model.generate_params()
    roa_file = "output/balance_roa/data.npz"
    roa_path = Path(__file__).resolve().parent / roa_file
    angles, angular_velocities, converged_grid = load_roa(roa_path)
    settings = {
        "velocity_range": [0.0, np.sqrt(2 * params["gravity"] / params["length"])],
        "velocity_points": 24,
        "angle_of_attack_range": [np.pi / 8, np.pi / 7],
        "angle_of_attack_points": 3,
        "timestep": 1e-4,
        "duration": 8.0,
    }
    initial_velocities = np.linspace(
        *settings["velocity_range"], settings["velocity_points"]
    )
    angle_of_attack_values = np.linspace(
        *settings["angle_of_attack_range"], settings["angle_of_attack_points"]
    )
    next_velocities, reached_roa, initially_in_roa = build_transition_table(
        initial_velocities,
        angle_of_attack_values,
        params,
        angles,
        angular_velocities,
        converged_grid,
        settings["timestep"],
        settings["duration"],
    )
    print_transition_table(
        initial_velocities,
        angle_of_attack_values,
        next_velocities,
        reached_roa,
        initially_in_roa,
    )
    minimum_steps, selected_angles = compute_minimum_step_policy(
        initial_velocities,
        angle_of_attack_values,
        next_velocities,
        reached_roa,
        initially_in_roa,
    )
    print_step_policy(initial_velocities, minimum_steps, selected_angles)
    maximum_steps, maximum_selected_angles = compute_maximum_step_policy(
        initial_velocities,
        angle_of_attack_values,
        next_velocities,
        reached_roa,
        initially_in_roa,
    )
    print_step_policy(
        initial_velocities, maximum_steps, maximum_selected_angles, "Maximum"
    )
    configuration = {"params": params, "roa_file": roa_file, **settings}
    output = save_transition_table(
        initial_velocities,
        angle_of_attack_values,
        next_velocities,
        reached_roa,
        initially_in_roa,
        minimum_steps,
        selected_angles,
        maximum_steps,
        maximum_selected_angles,
        configuration,
    )
    print(
        f"Saved {next_velocities.shape[0]} x {next_velocities.shape[1]} table: {output}"
    )
