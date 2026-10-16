"""Inspect the assignment's first full height-control policy."""

import argparse
import sys
import threading
from pathlib import Path

import torch

ASSIGNMENT_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT_REPO))

from scripts.play import get_play_args, setup  # noqa: E402
from gym.utils.interfaces.CommandVisualizer import CommandVisualizer  # noqa: E402
from gym.utils.interfaces.MujocoKeyboardInterface import MujocoKeyboardInterface  # noqa: E402

EXPERIMENT = "a4_height_full_v1"
RUN = "Oct15_01-26-07_"
CHECKPOINT = 550


def load_height(
    headless=False,
    *,
    experiment_name=EXPERIMENT,
    load_run=RUN,
    checkpoint=CHECKPOINT,
    seed=42,
):
    flags = [
        "--task=go2trot",
        "--device=cpu",
        "--num_envs=1",
        f"--experiment_name={experiment_name}",
        f"--checkpoint={checkpoint}",
        "--original_cfg",
        f"--seed={seed}",
    ]
    if load_run is not None:
        flags.append(f"--load_run={load_run}")
    if headless:
        flags.append("--headless")
    env, runner = setup(get_play_args(flags))
    assert env.commands.shape == (1, 4)
    assert runner.get_obs(runner.actor_cfg["obs"]).shape == (1, 50)
    keyboard = MujocoKeyboardInterface(env)
    assert env.cfg.commands.resampling_time == 1e9
    if not headless:
        CommandVisualizer(env)
    return env, runner, keyboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", default=EXPERIMENT)
    parser.add_argument("--load_run", default=RUN)
    parser.add_argument("--checkpoint", type=int, default=CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    starting_threads = set(threading.enumerate())
    with torch.no_grad():
        env, runner, _ = load_height(
            experiment_name=args.experiment_name,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
            seed=args.seed,
        )
        viewer = env._backend._viewer
        viewer_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in starting_threads
            and thread.name.endswith("(_launch_internal)")
        ]
        print(
            f"{args.experiment_name}: checkpoint {runner.it}, "
            f"playback seed {args.seed}."
        )
        print("Use Up/Down for height; I/K, J/L and N/M control velocity targets.")
        print("Close the viewer window or press Ctrl+C in the terminal to stop.")
        last_target = None
        try:
            while viewer.is_running() and not env.exit:
                target = env.commands[0, 3].item()
                if target != last_target:
                    print(
                        f"Target height: {target:.2f} m; "
                        f"current height: {env.base_height[0, 0].item():.3f} m",
                        flush=True,
                    )
                    last_target = target
                runner.set_actions(
                    runner.actor_cfg["actions"],
                    runner.get_inference_actions(),
                    runner.actor_cfg["disable_actions"],
                )
                env.step()
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()
            for thread in viewer_threads:
                thread.join()
            env._backend.close()


if __name__ == "__main__":
    main()
