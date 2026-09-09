"""Check rimless-wheel dynamics, continuous integration, and impacts."""

import numpy as np
from models import rimless_wheel as model
from simulate_wheel import simulate_continuous_step


def check_continuous_dynamics(parameters):
    """Check two known state derivatives."""
    upright_derivative = model.calculate_state_derivative(
        0.0, np.array([0.0, 2.0]), parameters
    )
    tilted_derivative = model.calculate_state_derivative(
        0.0, np.array([np.pi / 6, 0.0]), parameters
    )
    np.testing.assert_allclose(
        upright_derivative,
        [2.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tilted_derivative,
        [0.0, 4.905],
        atol=1e-12,
    )
    return (
        f"Dynamics: theta_dot={upright_derivative[0]:g} rad/s (expected 2); "
        f"omega_dot=[{upright_derivative[1]:g}, {tilted_derivative[1]:g}] rad/s^2 "
        "(expected [0, 4.905] at theta=0, pi/6)"
    )


def check_continuous_step(parameters):
    """Check contact speed, timestep convergence, and energy conservation."""
    step_start_angle, step_end_angle = model.calculate_contact_angles(parameters)
    initial_angular_velocity = 2.0
    initial_state = np.array([step_start_angle, initial_angular_velocity])
    expected_velocity = np.sqrt(
        initial_angular_velocity**2
        + 2
        * parameters["gravity"]
        / parameters["spoke_length"]
        * (np.cos(step_start_angle) - np.cos(step_end_angle))
    )
    velocity_errors = []
    for time_step in (0.02, 0.01, 0.005):
        _, states, reason = simulate_continuous_step(
            initial_state, parameters, time_step
        )
        assert reason == "contact"
        velocity_errors.append(abs(states[1, -1] - expected_velocity))

    assert velocity_errors[0] > velocity_errors[1] > velocity_errors[2]
    np.testing.assert_allclose(states[1, -1], expected_velocity, rtol=0, atol=1e-8)
    np.testing.assert_allclose(states[0, -1], step_end_angle, rtol=0, atol=1e-12)
    kinetic_energy, potential_energy = model.calculate_energy(states, parameters)
    total_energy = kinetic_energy + potential_energy
    np.testing.assert_allclose(total_energy, total_energy[0], rtol=0, atol=1e-8)
    return (
        f"Contact (dt=0.005 s): omega={states[1, -1]:.9f} rad/s "
        f"(expected {expected_velocity:.9f}); "
        f"max energy error={np.max(np.abs(total_energy - total_energy[0])):.2e} J\n"
        "Velocity errors at dt=[0.02, 0.01, 0.005] s: "
        + ", ".join(f"{error:.2e}" for error in velocity_errors)
        + " rad/s"
    )


def check_impact(parameters):
    """Check impact geometry and energy changes."""
    slope_angle = parameters["slope_angle"]
    spoke_length = parameters["spoke_length"]
    half_spoke_angle = np.pi / parameters["spoke_count"]
    observations = []
    for direction in (-1, 1):
        contact_angle = slope_angle + direction * half_spoke_angle
        state_before_contact = np.array([contact_angle, direction * 2.0])
        assert model.detect_contact(state_before_contact, parameters)
        state_after_contact = model.reset_after_contact(state_before_contact, parameters)
        assert not model.detect_contact(state_after_contact, parameters)
        np.testing.assert_allclose(
            state_after_contact[0], slope_angle - direction * half_spoke_angle
        )
        np.testing.assert_allclose(state_after_contact[1], direction * np.sqrt(2))

        # Reconstruct the hub position after changing the support foot.
        new_contact_position = (
            direction
            * 2
            * spoke_length
            * np.sin(half_spoke_angle)
            * np.array([np.cos(slope_angle), -np.sin(slope_angle)])
        )
        hub_before = spoke_length * np.array(
            [np.sin(contact_angle), np.cos(contact_angle)]
        )
        angle_after = state_after_contact[0]
        hub_after = new_contact_position + spoke_length * np.array(
            [np.sin(angle_after), np.cos(angle_after)]
        )
        np.testing.assert_allclose(hub_after, hub_before, atol=1e-12)

        kinetic_before, potential_before = model.calculate_energy(
            state_before_contact, parameters
        )
        kinetic_after, potential_after = model.calculate_energy(
            state_after_contact, parameters, contact_height=new_contact_position[1]
        )
        np.testing.assert_allclose(kinetic_after, kinetic_before / 2)
        np.testing.assert_allclose(potential_after, potential_before)
        observations.append(
            f"Impact ({state_before_contact[1]:+g} rad/s): "
            f"omega_after={state_after_contact[1]:.9f} rad/s "
            f"(expected {direction * np.sqrt(2):.9f}); "
            f"kinetic energy ratio={kinetic_after / kinetic_before:.6f} (expected 0.5)"
        )
    return "\n".join(observations)


def main():
    parameters = model.create_parameters()
    observations = [
        check_continuous_dynamics(parameters),
        check_continuous_step(parameters),
        check_impact(parameters),
    ]
    print("\n".join(observations))


if __name__ == "__main__":
    main()
