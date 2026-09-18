import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from controllers import compute_torque
from models import inverted_pendulum_walker as model


def simulate_balance(initial_states, params, timestep, duration):
    """Simulate balancing about one fixed support foot."""
    states = initial_states.copy()
    simulation_params = params.copy()
    minimum_angle = params["incline"] - np.pi / 2
    maximum_angle = params["incline"] + np.pi / 2
    fallen = (states[0] <= minimum_angle) | (states[0] >= maximum_angle)

    for step in range(round(duration / timestep)):
        simulation_params["ankle_torque"] = compute_torque(states, params)
        state_derivative = model.dynamics(step * timestep, states, simulation_params)
        state_derivative[:, fallen] = 0
        states += timestep * state_derivative
        # Stop trajectories when the hub reaches the ground.
        fallen |= (states[0] <= minimum_angle) | (states[0] >= maximum_angle)

    return states, fallen


def plot_roa(initial_states, converged):
    """Plot the initial states classified as converged."""
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.scatter(
        *initial_states[:, ~converged],
        s=1,
        linewidths=0,
        color="lightgray",
        label="Not confirmed",
    )
    ax.scatter(
        *initial_states[:, converged],
        s=1,
        linewidths=0,
        color="tab:blue",
        label="Converged",
    )
    ax.plot(0, 0, "o", color="red", markersize=5, label="Upright equilibrium")
    ax.set(
        xlabel="Initial angle (rad)",
        ylabel="Initial angular velocity (rad/s)",
        title="Standing controller: sampled region of attraction",
    )
    ax.legend()
    return fig


def save_results(fig, initial_states, final_states, converged, fallen, configuration):
    """Save the figure, data, configuration, and source files."""
    folder = Path(__file__).resolve().parent
    output = folder / "output" / "balance_roa"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "roa.png", dpi=150)
    np.savez(
        output / "data.npz",
        initial_states=initial_states,
        final_states=final_states,
        converged=converged,
        fallen=fallen,
    )
    (output / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    for relative_path in (
        "analyze_balance_roa.py",
        "controllers.py",
        "models/__init__.py",
        "models/inverted_pendulum_walker.py",
    ):
        target = output / "source" / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / relative_path, target)


def main():
    params = model.generate_params()
    params.update(angle_gain=1.0, angular_velocity_gain=2.0)
    settings = {
        "angle_range": [params["incline"] - np.pi / 7, params["incline"] + np.pi / 7],
        "angular_velocity_range": [-5.5, 5.5],
        "grid_points": 401,
        "timestep": 0.001,
        "duration": 30.0,
        "angle_tolerance": 0.001,
        "angular_velocity_tolerance": 0.001,
    }
    angles = np.linspace(*settings["angle_range"], settings["grid_points"])
    velocities = np.linspace(
        *settings["angular_velocity_range"], settings["grid_points"]
    )
    angle_grid, velocity_grid = np.meshgrid(angles, velocities)
    initial_states = np.array([angle_grid.ravel(), velocity_grid.ravel()])
    final_states, fallen = simulate_balance(
        initial_states, params, settings["timestep"], settings["duration"]
    )
    converged = (
        ~fallen
        & (np.abs(final_states[0]) < settings["angle_tolerance"])
        & (np.abs(final_states[1]) < settings["angular_velocity_tolerance"])
    )
    configuration = {"params": params, **settings}
    fig = plot_roa(initial_states, converged)
    save_results(fig, initial_states, final_states, converged, fallen, configuration)
    plt.show()


if __name__ == "__main__":
    main()
