"""Simulate rimless-wheel motion between repeated impacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from integrators import rk4
from models import rimless_wheel

EVENT_ANGLE_TOLERANCE = 1e-12  # rad
EVENT_TIME_TOLERANCE = 1e-12  # s
STANDING_VELOCITY_TOLERANCE = 1e-4  # rad/s


def locate_contact(time, state, time_step, parameters, direction):
    """Locate contact within a timestep that crosses the contact angle."""
    step_start_angle, step_end_angle = rimless_wheel.calculate_contact_angles(parameters)
    contact_angle = step_end_angle if direction > 0 else step_start_angle
    lower_duration = 0.0
    upper_duration = time_step

    for _ in range(60):
        trial_duration = (lower_duration + upper_duration) / 2
        trial_state = rk4.step(
            rimless_wheel.calculate_state_derivative,
            time,
            state,
            trial_duration,
            parameters,
        )
        angle_error = direction * (trial_state[0] - contact_angle)
        moving_outward = direction * trial_state[1] > 0

        if (
            abs(angle_error) <= EVENT_ANGLE_TOLERANCE
            and upper_duration - lower_duration <= EVENT_TIME_TOLERANCE
            and moving_outward
        ):
            return time + trial_duration, trial_state
        if angle_error < 0 or not moving_outward:
            lower_duration = trial_duration
        else:
            upper_duration = trial_duration

    raise RuntimeError("Contact localization did not reach the event tolerances.")


def simulate_continuous_step(initial_state, parameters, time_step=0.01, max_time=10.0):
    """Return local times, states, and the reason this continuous phase ended."""
    time = 0.0
    state = np.array(initial_state, dtype=float)
    times = [time]
    states = [state]

    if rimless_wheel.detect_contact(state, parameters):
        return np.array(times), np.array(states).T, "contact"

    while time < max_time:
        current_time_step = min(time_step, max_time - time)
        # Resolve departure before a possible return to the same boundary.
        if time == 0:
            angular_acceleration = rimless_wheel.calculate_state_derivative(
                time, state, parameters
            )[1]
            if state[1] * angular_acceleration < 0:
                current_time_step = min(
                    current_time_step, abs(state[1] / angular_acceleration)
                )
        next_state = rk4.step(
            rimless_wheel.calculate_state_derivative,
            time,
            state,
            current_time_step,
            parameters,
        )
        if not np.all(np.isfinite(next_state)):
            return np.array(times), np.array(states).T, "nonfinite_state"

        if rimless_wheel.detect_contact(next_state, parameters):
            try:
                contact_time, contact_state = locate_contact(
                    time, state, current_time_step, parameters, np.sign(next_state[1])
                )
            except RuntimeError:
                return (
                    np.array(times),
                    np.array(states).T,
                    "contact_localization_failed",
                )
            times.append(contact_time)
            states.append(contact_state)
            return np.array(times), np.array(states).T, "contact"

        time += current_time_step
        state = next_state
        times.append(time)
        states.append(state)

    return np.array(times), np.array(states).T, "time_limit"


def simulate_wheel(
    initial_state,
    parameters,
    number_of_steps=30,
    time_step=0.01,
    max_time=30.0,
    standing_velocity_tolerance=STANDING_VELOCITY_TOLERANCE,
):
    """Return times, states, and the reason the hybrid simulation ended."""
    state = np.array(initial_state, dtype=float)
    times = [0.0]
    states = [state]

    reason = "step_limit"
    for _ in range(number_of_steps):
        if rimless_wheel.detect_standing(
            states[-1], parameters, standing_velocity_tolerance
        ):
            reason = "standing"
            break
        remaining_time = max_time - times[-1]
        if remaining_time <= 0:
            reason = "time_limit"
            break
        step_times, step_states, reason = simulate_continuous_step(
            states[-1], parameters, time_step, remaining_time
        )
        times.extend(times[-1] + step_times[1:])
        states.extend(step_states[:, 1:].T)
        if reason != "contact":
            break

        state_after_contact = rimless_wheel.reset_after_contact(states[-1], parameters)
        times.append(times[-1])
        states.append(state_after_contact)
        reason = "step_limit"

    if reason == "step_limit" and rimless_wheel.detect_standing(
        states[-1], parameters, standing_velocity_tolerance
    ):
        reason = "standing"
    return np.array(times), np.array(states).T, reason


def plot_trajectory(times, states, parameters):
    """Save angle and angular velocity time histories."""
    impact_indices = np.flatnonzero(np.diff(times) == 0) + 1
    figure, axes = plt.subplots(
        2, 1, sharex=True, figsize=(11, 6), layout="constrained"
    )
    figure.suptitle(
        f"Rimless wheel: {len(impact_indices)} steps, "
        f"{parameters['spoke_count']} spokes, "
        f"slope = {np.rad2deg(parameters['slope_angle']):g} deg"
    )

    axes[0].plot(times, np.rad2deg(states[0]), color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel(r"Stance angle $\theta$ (deg)")

    axes[1].plot(times, states[1], color="tab:blue", linewidth=1.2)
    axes[1].scatter(
        times[impact_indices],
        states[1, impact_indices],
        color="tab:orange",
        s=16,
        label="After impact",
        zorder=3,
    )
    axes[1].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s)")
    axes[1].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(times[0], times[-1])

    figure_path = Path(__file__).parent / "figures" / "forward_trajectory.png"
    figure_path.parent.mkdir(exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def main():
    """Plot thirty steps and summarize the post-impact velocity."""
    parameters = rimless_wheel.create_parameters()
    initial_angular_velocity = 2.0  # rad/s
    step_start_angle, _ = rimless_wheel.calculate_contact_angles(parameters)
    initial_state = [step_start_angle, initial_angular_velocity]
    number_of_steps = 30
    times, states, reason = simulate_wheel(initial_state, parameters, number_of_steps)
    plot_trajectory(times, states, parameters)
    impact_indices = np.flatnonzero(np.diff(times) == 0) + 1
    print(
        f"{reason}: {len(impact_indices)} impacts, time={times[-1]:.6f} s, "
        f"angular velocity={states[1, -1]:.6f} rad/s"
    )


if __name__ == "__main__":
    main()
