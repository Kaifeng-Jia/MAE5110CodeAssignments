"""Fourth-order Runge-Kutta integration, adapted from assignment 0."""


def step(calculate_derivative, time, state, time_step, parameters):
    """Advance the state by one RK4 step."""
    initial_derivative = calculate_derivative(time, state, parameters)

    first_midpoint_derivative = calculate_derivative(
        time + time_step / 2,
        state + time_step / 2 * initial_derivative,
        parameters,
    )

    second_midpoint_derivative = calculate_derivative(
        time + time_step / 2,
        state + time_step / 2 * first_midpoint_derivative,
        parameters,
    )

    final_derivative = calculate_derivative(
        time + time_step,
        state + time_step * second_midpoint_derivative,
        parameters,
    )

    next_state = state + time_step / 6 * (
        initial_derivative
        + 2 * first_midpoint_derivative
        + 2 * second_midpoint_derivative
        + final_derivative
    )
    return next_state
