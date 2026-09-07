# Assignment 0

[Instructor prompt](assignment.md)

Run these commands from the repository root:

```bash
uv sync --python 3.14
uv run python assignments/assignment_0/assignment_0.py
uv run python assignments/assignment_0/compare_euler_rk4.py
uv run python assignments/assignment_0/simulate_bouncing_ball.py
```

- `assignment_0.py`: pendulum simulation and energy plot.
- `compare_euler_rk4.py`: timing comparison of Euler and RK4.
- `simulate_bouncing_ball.py`: bouncing-ball simulation and energy plot.
- `models/` and `integrators/`: the implementations used by this assignment.
- `figures/`: nine existing Matplotlib images moved from the outer workspace.

The reorganization preserves the source code, experiment parameters, image
contents, and image filenames. Existing images were not regenerated; their
filenames alone do not establish the parameters used to produce them.
`Figure_1.png` retains its original name until its experiment is identified.

The simulation scripts display plots with `plt.show()`. Running them does not
automatically overwrite the archived images in `figures/`.
