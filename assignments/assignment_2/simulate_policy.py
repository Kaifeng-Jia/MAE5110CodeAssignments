import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from controllers import is_in_roa, load_roa
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from poincare_map import simulate_return


def simulate_policy(
    initial_velocity,
    initial_velocities,
    selected_angles,
    params,
    angles,
    angular_velocities,
    converged_grid,
    timestep=1e-4,
    duration=8.0,
    step_limit=12,
):
    """Follow the lookup policy using actual simulated return velocities."""
    velocity = initial_velocity
    history = []
    states = [[0.0, velocity]]
    impact_indices = []
    reached_roa = is_in_roa(states[0], angles, angular_velocities, converged_grid)
    for _ in range(step_limit):
        if reached_roa:
            break
        state_index = np.argmin(np.abs(initial_velocities - velocity))
        alpha = selected_angles[state_index]
        if np.isnan(alpha):
            break

        next_velocity, reached_roa, step_trajectory = simulate_return(
            velocity,
            alpha,
            params,
            angles,
            angular_velocities,
            converged_grid,
            timestep,
            duration,
        )
        # Join at the shared section state; retain both sides of each impact.
        impact_indices.extend(step_trajectory["impact_indices"] + len(states) - 1)
        states.extend(step_trajectory["states"][1:])
        history.append(
            {
                "velocity": velocity,
                "lookup_velocity": initial_velocities[state_index],
                "alpha": alpha,
                "next_velocity": next_velocity,
                "reached_roa": reached_roa,
            }
        )
        if reached_roa or next_velocity is None:
            break
        velocity = next_velocity

    trajectory = {
        "states": np.array(states),
        "impact_indices": np.array(impact_indices, dtype=int),
    }
    return history, bool(reached_roa), trajectory


def plot_comparison(rollouts, angles, angular_velocities, converged_grid):
    """Compare continuous paths and impact resets against the estimated RoA."""
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    roa_cells = (
        converged_grid[:-1, :-1]
        & converged_grid[1:, :-1]
        & converged_grid[:-1, 1:]
        & converged_grid[1:, 1:]
    )
    roa_color = "#dcebd5"
    ax.pcolormesh(
        angles,
        angular_velocities,
        np.ma.masked_where(~roa_cells, roa_cells),
        cmap=ListedColormap([roa_color]),
        shading="flat",
    )

    # Different widths keep both colors visible on shared paths.
    for (name, rollout), color, width in zip(
        rollouts.items(), ("tab:blue", "tab:orange"), (3.2, 1.5)
    ):
        states = rollout["trajectory"]["states"]
        impact_indices = rollout["trajectory"]["impact_indices"]
        segments = np.split(states, impact_indices)
        for index, segment in enumerate(segments):
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linewidth=width,
                label=name.capitalize() if index == 0 else None,
            )
        for index in impact_indices:
            ax.plot(
                states[index - 1 : index + 1, 0],
                states[index - 1 : index + 1, 1],
                "--",
                color=color,
                linewidth=width,
            )
        ax.plot(*states[-1], "o", color=color, markersize=7, zorder=5)

    initial_state = next(iter(rollouts.values()))["trajectory"]["states"][0]
    ax.plot(*initial_state, "ko", markersize=5, zorder=6)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=roa_color, label="RoA"))
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    roa_velocities = angular_velocities[converged_grid.any(axis=1)]
    maximum_velocity = max(
        rollout["trajectory"]["states"][:, 1].max() for rollout in rollouts.values()
    )
    ax.set(
        xlim=(angles[0] - 0.03, angles[-1] + 0.03),
        ylim=(roa_velocities.min() - 0.2, maximum_velocity + 0.3),
        xlabel=r"Angle $\theta$ (rad)",
        ylabel=r"Angular velocity $\omega$ (rad/s)",
        title="Minimum- and maximum-step policies",
    )
    ax.grid(alpha=0.2)
    return fig


def print_rollout(initial_velocity, policy_name, predicted_steps, history, reached_roa):
    """Print the policy prediction and simulated steps."""
    print(
        f"\nInitial velocity: {initial_velocity:.4f} rad/s; "
        f"{policy_name} policy predicted steps: {predicted_steps}"
    )
    print("step | actual velocity | lookup velocity | alpha (deg) | result")
    for step, record in enumerate(history, start=1):
        if record["reached_roa"]:
            result = "Reached RoA"
        elif record["next_velocity"] is None:
            result = "No return"
        else:
            result = f"{record['next_velocity']:.4f} rad/s"
        print(
            f"{step:>4} | {record['velocity']:>15.4f} | "
            f"{record['lookup_velocity']:>15.4f} | "
            f"{np.rad2deg(record['alpha']):>11.2f} | {result}"
        )
    print(f"Simulated step attempts: {len(history)}; reached_roa = {reached_roa}")


def save_comparison(rollouts, configuration, figure):
    """Save both trajectories, histories, settings, and figure."""
    folder = Path(__file__).resolve().parent
    output = folder / "output" / "policy_comparison"
    output.mkdir(parents=True, exist_ok=True)
    results = {
        name: {key: value for key, value in rollout.items() if key != "trajectory"}
        for name, rollout in rollouts.items()
    }
    (output / "results.json").write_text(
        json.dumps({"configuration": configuration, "policies": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    np.savez(
        output / "trajectories.npz",
        **{
            f"{name}_{key}": value
            for name, rollout in rollouts.items()
            for key, value in rollout["trajectory"].items()
        },
    )
    figure.savefig(output / "phase_comparison.png", dpi=200)
    return output


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    policy_folder = "output/transition_table"
    configuration = json.loads((folder / policy_folder / "config.json").read_text())
    with np.load(folder / policy_folder / "data.npz") as data:
        initial_velocities = data["initial_velocities"]
        policies = (
            ("minimum", data["selected_angles"], data["minimum_steps"]),
            ("maximum", data["maximum_selected_angles"], data["maximum_steps"]),
        )
    angles, angular_velocities, converged_grid = load_roa(
        folder / configuration["roa_file"]
    )
    # Keep the same initial state when changing the grid.
    initial_velocity = 4
    initial_index = np.argmin(np.abs(initial_velocities - initial_velocity))
    step_limit = 12
    rollouts = {}
    for policy_name, selected_angles, step_counts in policies:
        predicted_steps = int(step_counts[initial_index])
        history, reached_roa, trajectory = simulate_policy(
            initial_velocity,
            initial_velocities,
            selected_angles,
            configuration["params"],
            angles,
            angular_velocities,
            converged_grid,
            configuration["timestep"],
            configuration["duration"],
            step_limit,
        )
        print_rollout(
            initial_velocity, policy_name, predicted_steps, history, reached_roa
        )
        rollouts[policy_name] = {
            "predicted_steps": predicted_steps,
            "reached_roa": reached_roa,
            "history": history,
            "trajectory": trajectory,
        }
    run_configuration = {
        **configuration,
        "policy_folder": policy_folder,
        "initial_velocity": initial_velocity,
        "step_limit": step_limit,
    }
    figure = plot_comparison(rollouts, angles, angular_velocities, converged_grid)
    output = save_comparison(rollouts, run_configuration, figure)
    print(f"Saved {output}")
    plt.show()
