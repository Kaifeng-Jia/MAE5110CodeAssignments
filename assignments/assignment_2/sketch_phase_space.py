from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from controllers import load_roa
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from models import inverted_pendulum_walker as model


def draw_flow(ax, angles, velocities, color, width=2):
    """Draw a continuous trajectory with its direction of motion."""
    ax.plot(angles, velocities, color=color, linewidth=width)
    middle = len(angles) // 2
    ax.annotate(
        "",
        xy=(angles[middle + 12], velocities[middle + 12]),
        xytext=(angles[middle - 12], velocities[middle - 12]),
        arrowprops={"arrowstyle": "->", "color": color, "lw": width},
    )


def plot_phase_sketch(params, roa):
    """Sketch passive flow, contact resets, the section, and backward failure."""
    fig, (ax, failure_ax) = plt.subplots(
        1, 2, figsize=(12, 6), width_ratios=(2.3, 1), layout="constrained"
    )
    gravity = params["gravity"]
    length = params["length"]
    incline = params["incline"]
    initial_velocity = 2.0
    angles, velocities, converged = roa
    cells = (
        converged[:-1, :-1]
        & converged[1:, :-1]
        & converged[:-1, 1:]
        & converged[1:, 1:]
    )
    ax.pcolormesh(
        angles,
        velocities,
        np.ma.masked_where(~cells, cells),
        cmap=ListedColormap(["#dcebd5"]),
        shading="flat",
    )
    ax.vlines(0, 0, 3.2, colors="black", linestyles="-.", linewidth=1.3)
    ax.annotate(
        "Poincaré section\n" + r"$\theta=0,\;\omega>0$",
        xy=(0, 3.0),
        xytext=(-0.4, 3.0),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=11,
    )

    for alpha, color, width, pose in zip(
        (np.pi / 8, np.pi / 7), ("tab:blue", "tab:orange"), (3, 1.6), (2, 3)
    ):
        contact_angle = incline + alpha
        reset_angle = incline - alpha
        theta = np.linspace(0, contact_angle, 200)
        omega = np.sqrt(
            initial_velocity**2 + 2 * gravity / length * (1 - np.cos(theta))
        )
        draw_flow(ax, theta, omega, color, width)
        pre_impact = (contact_angle, omega[-1])
        post_impact = (reset_angle, omega[-1] * np.cos(2 * alpha))
        ax.vlines(contact_angle, 0, 3.2, colors=color, linestyles=":", linewidth=1.5)
        ax.text(
            contact_angle + 0.008,
            0.2,
            rf"$\theta^- = \gamma + \pi/{8 if pose == 2 else 7}$",
            rotation=90,
            color=color,
            fontsize=11,
        )
        ax.annotate(
            "",
            xy=post_impact,
            xytext=pre_impact,
            arrowprops={
                "arrowstyle": "->",
                "linestyle": "--",
                "color": color,
                "lw": 1.6,
            },
        )
        ax.plot(*pre_impact, "o", color=color, markersize=6)
        ax.plot(*post_impact, "o", color=color, markerfacecolor="white", markersize=6)
        ax.annotate(
            str(pose),
            pre_impact,
            xytext=(-14, 10) if pose == 2 else (8, 9),
            textcoords="offset points",
            color=color,
            fontsize=13,
            fontweight="bold",
        )
        theta = np.linspace(reset_angle, 0, 200)
        omega = np.sqrt(
            post_impact[1] ** 2
            + 2 * gravity / length * (np.cos(reset_angle) - np.cos(theta))
        )
        draw_flow(ax, theta, omega, color, width)
        ax.plot(0, omega[-1], "o", color=color, markersize=4)

    ax.plot(0, initial_velocity, "ko", markersize=5)
    ax.annotate(
        "1",
        (0, initial_velocity),
        xytext=(-15, 8),
        textcoords="offset points",
        fontsize=13,
        fontweight="bold",
    )
    ax.plot(0, 0, "o", color="red", markersize=5)
    ax.text(
        -0.43,
        -0.72,
        "Dashed arrows: impact reset\n"
        + r"$\theta^+=\gamma-\alpha$"
        + "\n"
        + r"$\omega^+=\omega^-\cos(2\alpha)$",
        fontsize=11,
        va="top",
    )
    ax.legend(
        handles=[
            Line2D([], [], color="tab:blue", label=r"$\alpha=\pi/8$"),
            Line2D([], [], color="tab:orange", label=r"$\alpha=\pi/7$"),
            Patch(facecolor="#dcebd5", label="Estimated RoA"),
        ],
        loc="upper left",
        bbox_to_anchor=(0, 0.88),
        fontsize=10,
    )
    ax.set(
        xlim=(-0.47, 0.62),
        ylim=(-1.5, 3.4),
        xlabel=r"Angle $\theta$ (rad)",
        ylabel=r"Angular velocity $\omega$ (rad/s)",
        title=r"Walking: $\omega_0=2$ rad/s, $\tau=0$",
    )

    ground_angle = incline - np.pi / 2
    theta = np.linspace(-0.33, ground_angle, 200)
    omega = -np.sqrt(0.2**2 + 2 * gravity / length * (np.cos(theta[0]) - np.cos(theta)))
    draw_flow(failure_ax, theta, omega, "#6b5b73")
    failure_ax.axvline(ground_angle, color="gray", linestyle=":")
    failure_ax.plot(theta[-1], omega[-1], "o", color="#6b5b73", markersize=6)
    failure_ax.annotate(
        "4  Hub reaches ground",
        (theta[-1], omega[-1]),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10,
    )
    failure_ax.text(-1.52, 0.2, r"$\theta=\gamma-\pi/2$", fontsize=11)
    failure_ax.set(
        xlim=(-1.65, -0.1),
        ylim=(-4.8, 0.6),
        xlabel=r"Angle $\theta$ (rad)",
        ylabel=r"Angular velocity $\omega$ (rad/s)",
        title=r"Backward fall: $\tau=0$",
    )
    for axis in (ax, failure_ax):
        axis.grid(alpha=0.15)
    return fig


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    figure = plot_phase_sketch(
        model.generate_params(), load_roa(folder / "output/balance_roa/data.npz")
    )
    output = folder / "output" / "phase_sketch"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "phase_sketch.png", dpi=200)
    figure.savefig(output / "phase_sketch.pdf")
    plt.show()
