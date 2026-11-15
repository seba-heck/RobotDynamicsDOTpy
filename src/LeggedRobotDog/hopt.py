
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space


def hopt(A, b, C, d, print_solution=False):
    """
    Naive hierarchical least-squares optimization.

    - Tasks are processed in priority order.
    - Successful tasks are preserved.
    - Failed tasks are ignored.
    - No equality constraints are passed to SLSQP.
    - One-dimensional problems are solved analytically.
    """

    A = [np.asarray(Ai, dtype=float) for Ai in A]
    b = [np.asarray(bi, dtype=float).reshape(-1) for bi in b]

    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)

    n = A[0].shape[1]

    x = np.zeros(n)

    # Previously accepted task constraints
    Aeq = np.empty((0, n))
    beq = np.empty(0)

    feasibility_tol = 1e-6

    for task_idx, (Ai, bi) in enumerate(zip(A, b)):

        # print(f"[HOPT] Task {task_idx}")

        try:
            # Compute nullspace of previous tasks
            if Aeq.shape[0] == 0:
                x_base = x.copy()
                N = np.eye(n)
            else:
                x_base = np.linalg.lstsq(Aeq, beq, rcond=None)[0]
                N = null_space(Aeq)

            free_variables = N.shape[1]

            # print(
            #     f"[HOPT] Task {task_idx}: "
            #     f"free variables={free_variables}"
            # )

            # if free_variables == 0:
            #     print(f"[HOPT] Task {task_idx}: no freedom left. Skipping.")
            #     continue

            # Transform task into reduced coordinates
            AN = Ai @ N
            residual = Ai @ x_base - bi

            # Transform inequalities
            Cr = C @ N
            dr = d - C @ x_base

            # Initial reduced solution
            z0 = N.T @ (x - x_base)

            # -------------------------------------------------
            # One-dimensional optimization
            # -------------------------------------------------
            if free_variables == 1:

                lower = -np.inf
                upper = np.inf

                feasible = True

                for ci, di in zip(Cr[:, 0], dr):

                    if abs(ci) < 1e-12:
                        if di < -feasibility_tol:
                            feasible = False
                            break
                        continue

                    bound = di / ci

                    if ci > 0:
                        upper = min(upper, bound)
                    else:
                        lower = max(lower, bound)

                if not feasible or lower > upper + feasibility_tol:
                    # print(
                    #     f"[HOPT] Task {task_idx}: "
                    #     "infeasible 1D problem. Skipping."
                    # )
                    continue

                # Handle tiny numerical interval inconsistencies
                if lower > upper:
                    midpoint = 0.5 * (lower + upper)
                    lower = midpoint
                    upper = midpoint

                denominator = float(AN.T @ AN)

                if denominator > 1e-14:
                    z_star = -float(AN.T @ residual) / denominator
                else:
                    z_star = float(z0[0])

                z_star = np.clip(z_star, lower, upper)
                z_star = np.array([z_star])

            # -------------------------------------------------
            # Multi-dimensional optimization
            # -------------------------------------------------
            else:

                def objective(z):
                    error = AN @ z + residual
                    return 0.5 * np.dot(error, error)

                def objective_jac(z):
                    return AN.T @ (AN @ z + residual)

                constraints = []

                if Cr.shape[0] > 0:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda z: dr - Cr @ z,
                        "jac": lambda z: -Cr,
                    })

                result = minimize(
                    objective,
                    z0,
                    jac=objective_jac,
                    method="SLSQP",
                    constraints=constraints,
                    options={
                        "maxiter": 100,
                        "ftol": 1e-8,
                        "disp": False,
                    },
                )

                # if not result.success:
                #     print(
                #         f"[HOPT] Task {task_idx} failed "
                #         f"({result.message}). Skipping."
                #     )
                #     continue

                z_star = result.x

            # Reconstruct solution
            candidate = x_base + N @ z_star

            # Check candidate
            if not np.all(np.isfinite(candidate)):
                # print(f"[HOPT] Task {task_idx}: invalid solution. Skipping.")
                continue

            if C.shape[0] > 0:
                violation = np.max(C @ candidate - d)

                if violation > feasibility_tol:
                    # print(
                    #     f"[HOPT] Task {task_idx}: "
                    #     f"constraint violation={violation:.3e}. Skipping."
                    # )
                    continue

            # Accept task
            x = candidate

            Aeq = np.vstack((Aeq, Ai))
            beq = np.concatenate((beq, Ai @ x))

            # print(f"[HOPT] Task {task_idx} accepted.")

        except Exception as error:
            # print(
            #     f"[HOPT] Task {task_idx} exception: {error}. Skipping."
            # )
            continue

    if print_solution:
        print("[HOPT] Final solution:")
        print(x)

    return x.reshape(-1, 1)
