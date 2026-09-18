from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from controllers import compute_torque, is_in_roa, load_roa
from matplotlib.animation import FuncAnimation, PillowWriter
from models import inverted_pendulum_walker as model

params = model.generate_params()
params["angle_gain"] = 1.0
params["angular_velocity_gain"] = 2.0
initial_state = np.array([0.0, 0.3])
timestep = 1e-4
sim_time = 8.0

roa_path = Path(__file__).resolve().parent / "output/balance_roa/data.npz"
angles, angular_velocities, converged_grid = load_roa(roa_path)


n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0
balancing = False

for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]
    if not balancing:
        balancing = is_in_roa(state, angles, angular_velocities, converged_grid)
    params["ankle_torque"] = compute_torque(state, params) if balancing else 0.0
    next_state = state + timestep * model.dynamics(t, state, params)

    if not balancing and model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(next_state, params)
        completed_steps += 1

    state_traj[:, step + 1] = next_state

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path(__file__).resolve().parent / "output"
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()
