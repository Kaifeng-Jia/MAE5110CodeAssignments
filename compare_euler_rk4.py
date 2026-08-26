import timeit

import numpy as np

from integrators import explicit_euler, rk4
from models import pendulum as model


params = {
    "gravity": 9.81,
    "length": 1,
    "mass": 0.2,
    "damping_coeff": 0.0,
}

initial_state = np.array([np.pi / 4, 0.0])
sim_time = 5.0


def run_simulation(integrator, timestep):
    """Run the pendulum simulation without plotting."""
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator.step(
            model.dynamics,
            t,
            state_traj[:, step],
            timestep,
            params,
        )

    return state_traj


def measure(integrator, timestep, number):
    """Return the average time for one simulation."""
    total_time = timeit.timeit(
        lambda: run_simulation(integrator, timestep),
        number=number,
    )
    return total_time / number


def main():
    number = 10
    benchmark_cases = [
        ("Explicit Euler, dt=1e-3", explicit_euler, 1e-3),
        ("RK4, dt=1e-3", rk4, 1e-3),
        ("RK4, dt=1e-1", rk4, 1e-1),
    ]

    for label, integrator, timestep in benchmark_cases:
        average_time = measure(integrator, timestep, number)
        print(f"{label}: {average_time:.6f} seconds")


if __name__ == "__main__":
    main()
