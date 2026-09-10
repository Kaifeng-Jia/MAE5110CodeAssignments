"""Compare regions of attraction and convergence for six to twelve spokes."""

import analyze_return_map
import analyze_roa
import numpy as np
from models import rimless_wheel
from sweep_slope import run_sweep


def create_configuration(spoke_count):
    """Configure one spoke count with a longer observation period."""
    configuration = analyze_roa.create_configuration()
    parameters = configuration["parameters"]
    parameters["spoke_count"] = spoke_count
    step_start_angle, step_end_angle = rimless_wheel.calculate_contact_angles(parameters)
    configuration["scope"] = "Spoke-count comparison at a 15-degree slope"
    configuration["prediction"] = (
        "Standing basin shrinks and loses stability at twelve spokes"
    )
    configuration["grid"].update(
        angle_min=step_start_angle, angle_max=step_end_angle
    )
    configuration["simulation"].update(number_of_steps=60, max_time=60.0)
    configuration["simulation_limit_change"] = {
        "previous_number_of_steps": 30,
        "previous_max_time": 30.0,
        "reason": "More spokes give slower convergence per impact",
    }

    return_map = analyze_return_map.create_configuration()
    return_map["parameters"] = parameters
    return_map["initial_state"] = [step_start_angle, 2.0]
    return_map["simulation"] = configuration["simulation"].copy()
    return_map["prediction"] = "Rolling speed increases with spoke count"
    predicted_multiplier = np.cos(2 * np.pi / spoke_count) ** 2
    return_map["floquet"]["prediction"] = f"Local slope near {predicted_multiplier:.6f}"
    configuration["return_map"] = return_map
    return configuration


def main():
    """Compare six to twelve spokes at a fixed slope."""
    configurations = {
        f"spokes_{spoke_count:02d}": create_configuration(spoke_count)
        for spoke_count in range(6, 13)
    }
    run_sweep("spoke_sweep", configurations)


if __name__ == "__main__":
    main()
