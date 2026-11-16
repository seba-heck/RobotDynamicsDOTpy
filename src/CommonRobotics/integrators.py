import numpy as np


def step_euler(q,dq,dt, dynamics):
    """Explicit Euler step."""
    ddq = dynamics(q,dq)

    q_new = q + dt*dq
    dq_new = dq + dt*ddq

    return q_new,dq_new,ddq,dt

def step_semi_implicit_euler(q,dq,dt,dynamics):
    """Semi-implicit / sympletic Euler step."""
    ddq = dynamics(q,dq)

    dq_new = dq + dt*ddq
    q_new = q + dt*dq_new

    return q_new,dq_new,ddq,dt

def step_rk4(q, dq, dt, dynamics, nq=10):
    """Classical fourth-order Runge-Kutta."""
    # State vector x = [q, dq]
    x = np.concatenate([q, dq])

    def f(x):
        q = x[:nq]
        dq = x[nq:]
        ddq = dynamics(q, dq)
        return np.concatenate([dq, ddq])

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)

    x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    q_new = x_new[:nq]
    dq_new = x_new[nq:]

    # Acceleration corresponding to the new state
    ddq_new = dynamics(q_new, dq_new)

    return q_new, dq_new, ddq_new, dt

def step_adaptive_rk4(q, dq, dt, dynamics, tol=1e-6):
    while True:
        # 1 großer Schritt
        q1, dq1, ddq1, _ = step_rk4(q, dq, dt, dynamics)

        # 2 kleine Schritte
        qh, dqh, ddqh, _ = step_rk4(q, dq, dt/2, dynamics)
        q2, dq2, ddq2, _ = step_rk4(qh, dqh, dt/2, dynamics)

        # Fehlerabschätzung
        err = max(
            np.linalg.norm(q2 - q1, np.inf),
            np.linalg.norm(dq2 - dq1, np.inf)
        )

        if err < tol:
            dt_new = dt * min(2.0, 0.9 * (tol / max(err, 1e-16))**0.2)
            ddq = dynamics(q2, dq2).reshape(-1)
            return q2, dq2, ddq, dt_new

        # Schritt war zu groß -> verkleinern
        dt *= max(0.1, 0.9 * (tol / err)**0.2)

def step_adaptive_semi_implicit_euler(q, dq, dt, dynamics, tol=1e-4):
    while True:
        # 1 großer Schritt
        q1, dq1, ddq1, _ = step_semi_implicit_euler(q, dq, dt, dynamics)

        # 2 kleine Schritte
        qh, dqh, ddqh, _ = step_semi_implicit_euler(q, dq, dt/2, dynamics)
        q2, dq2, ddq2, _ = step_semi_implicit_euler(qh, dqh, dt/2, dynamics)

        # Fehlerabschätzung
        err = max(
            np.linalg.norm(q2 - q1, np.inf),
            np.linalg.norm(dq2 - dq1, np.inf)
        )

        if err < tol:
            dt_new = dt * min(2.0, 0.9 * (tol / max(err, 1e-16))**0.2)
            ddq = dynamics(q2, dq2).reshape(-1)
            return q2, dq2, ddq, dt_new

        # Schritt war zu groß -> verkleinern
        dt *= max(0.1, 0.9 * (tol / err)**0.2)


ALL_INTEGRATOR_STEPS = {
    "euler": step_euler,
    "semi_implicit_euler": step_semi_implicit_euler,
    "rk4": step_rk4,
    "adaptive_rk4": step_adaptive_rk4,
    "adaptive_semi_implicit_euler": step_adaptive_semi_implicit_euler,
}