def step(dynamics, t, state, timestep, params):
    state_next = state + timestep * dynamics(t, state, params)
    return state_next