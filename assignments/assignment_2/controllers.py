import numpy as np


def compute_torque(state, params):
    """Calculate the bounded balancing torque."""
    angle, angular_velocity = state
    mass = params["mass"]
    length = params["length"]
    gravity = params["gravity"]
    angle_gain = params["angle_gain"]
    angular_velocity_gain = params["angular_velocity_gain"]

    torque_scale = mass * gravity * length
    control_input = -torque_scale * np.sin(angle) - mass * length**2 * (
        angle_gain * angle + angular_velocity_gain * angular_velocity
    )
    return np.clip(control_input, -0.1 * torque_scale, 0.05 * torque_scale)


def load_roa(path):
    """Read the sampled RoA axes and classification grid."""
    with np.load(path) as data:
        angles = np.unique(data["initial_states"][0])
        angular_velocities = np.unique(data["initial_states"][1])
        converged_grid = data["converged"].reshape(len(angular_velocities), len(angles))
    return angles, angular_velocities, converged_grid


def is_in_roa(state, angles, angular_velocities, converged_grid):
    """Check whether all four corners of the containing cell converged."""
    angle, angular_velocity = state

    if not (
        angles[0] <= angle <= angles[-1]
        and angular_velocities[0] <= angular_velocity <= angular_velocities[-1]
    ):
        return False

    angle_index = np.searchsorted(angles[1:-1], angle)
    velocity_index = np.searchsorted(angular_velocities[1:-1], angular_velocity)

    return converged_grid[
        velocity_index : velocity_index + 2,
        angle_index : angle_index + 2,
    ].all()


def compute_minimum_step_policy(
    initial_velocities,
    angle_of_attack_values,
    next_velocities,
    reached_roa,
    initially_in_roa,
):
    """Find minimum step counts and landing angles on the sampled table."""
    minimum_steps = np.full(len(initial_velocities), np.inf)
    selected_angles = np.full(len(initial_velocities), np.nan)
    minimum_steps[initially_in_roa] = 0

    # Seed one-step states from simulated RoA entry.
    for state_index in range(len(initial_velocities)):
        if initially_in_roa[state_index]:
            continue
        successful_actions = np.flatnonzero(reached_roa[state_index])
        if successful_actions.size:
            minimum_steps[state_index] = 1
            selected_angles[state_index] = angle_of_attack_values[successful_actions[0]]

    for step_count in range(2, len(initial_velocities) + 1):
        for state_index in range(len(initial_velocities)):
            if np.isfinite(minimum_steps[state_index]):
                continue
            for action_index, next_velocity in enumerate(next_velocities[state_index]):
                if np.isnan(next_velocity):
                    continue
                next_index = np.argmin(np.abs(initial_velocities - next_velocity))
                if minimum_steps[next_index] == step_count - 1:
                    minimum_steps[state_index] = step_count
                    selected_angles[state_index] = angle_of_attack_values[action_index]
                    break

    return minimum_steps, selected_angles


def compute_maximum_step_policy(
    initial_velocities,
    angle_of_attack_values,
    next_velocities,
    reached_roa,
    initially_in_roa,
):
    """Find longest stopping paths when returns map to lower-speed rows."""
    maximum_steps = np.full(len(initial_velocities), -np.inf)
    selected_angles = np.full(len(initial_velocities), np.nan)
    maximum_steps[initially_in_roa] = 0

    for state_index in range(len(initial_velocities)):
        if initially_in_roa[state_index]:
            continue
        for action_index, next_velocity in enumerate(next_velocities[state_index]):
            if reached_roa[state_index, action_index]:
                candidate_steps = 1
            elif np.isfinite(next_velocity):
                next_index = np.argmin(np.abs(initial_velocities - next_velocity))
                assert next_index < state_index, (
                    "Maximum-step table needs lower-speed returns."
                )
                if maximum_steps[next_index] <= 0:
                    continue
                candidate_steps = 1 + maximum_steps[next_index]
            else:
                continue
            if candidate_steps > maximum_steps[state_index]:
                maximum_steps[state_index] = candidate_steps
                selected_angles[state_index] = angle_of_attack_values[action_index]

    return maximum_steps, selected_angles
