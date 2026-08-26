def step(dynamics, t, state, timestep, params):
    h = timestep
    k1 = dynamics(t, state, params)
    k2 = dynamics(t + h/2, state + h/2*k1, params)
    k3 = dynamics(t + h/2, state + h/2*k2, params)
    k4 = dynamics(t + h, state + h*k3, params)
    state_next = state + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
    return state_next