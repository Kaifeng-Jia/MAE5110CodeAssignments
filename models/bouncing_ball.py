import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    velocity = state[1]

    acceleration = -gravity

    state_derivative = np.array([velocity, acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "mass": 1, 
        "restitution": 0.8,
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity ** 2
    potential_energy = mass * gravity * height
    return kinetic_energy, potential_energy

def resolve_collision(state, params):
    height = state[0]
    velocity = state[1]
    restitution = params["restitution"]
    if height <= 0 and velocity < 0:
        height = 0
        velocity = -restitution * velocity
    state_after_collision = np.array([height, velocity])
    return state_after_collision