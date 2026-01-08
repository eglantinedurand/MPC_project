import numpy as np
import casadi as ca
from typing import Tuple


def rk4(f, x, u, h):
    """One RK4 step for continuous-time dynamics x_dot = f(x,u)."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * h * k1, u)
    k3 = f(x + 0.5 * h * k2, u)
    k4 = f(x + h * k3, u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class NmpcCtrl:
    """
    Nonlinear MPC controller for landing.

    get_u provides:
        u0, x_ol, u_ol, t_ol = nmpc.get_u(t0, x0)
    with:
        x_ol shape: (12, N+1)
        u_ol shape: (4, N)
        t_ol shape: (N+1,)
    """

    def __init__(self, rocket, H: float, xs: np.ndarray, us: np.ndarray):
        """
        Constructor matches the notebook call:
            nmpc = NmpcCtrl(rocket, H, xs, us)

        rocket.f_symbolic(x,u) gives continuous-time x_dot = f(x,u),
        so we discretize internally with RK4 (as in the class NMPC exercise).
        """
        self.rocket = rocket
        self.Ts = float(getattr(rocket, "Ts", 1 / 20))
        self.H = float(H)
        self.N = int(np.round(self.H / self.Ts))
        self.H = self.N * self.Ts  # snap to integer number of steps

        # continuous-time symbolic dynamics
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]

        # trim point provided by notebook
        self.xs = np.asarray(xs, dtype=float).reshape(12,)
        self.us = np.asarray(us, dtype=float).reshape(4,)

        # keep a reference for tracking (use trim state)
        self.x_ref = self.xs.copy()
        self.u_ref = self.us.copy()

        # warm-start memory
        self._X_last = None
        self._U_last = None

        self._setup_controller()

    def _setup_controller(self) -> None:
        nx, nu, N = 12, 4, self.N

        # discrete dynamics map (RK4)
        def f_d(x, u):
            return rk4(self.f, x, u, self.Ts)

        opti = ca.Opti()

        # decision variables
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)

        # parameter: current state
        X0 = opti.parameter(nx, 1)

        # initial condition
        opti.subject_to(X[:, [0]] == X0)

        # ---- constraints (project + reasonable actuator bounds) ----
        dmax = 0.26          # +/- 15deg
        pdiff_max = 20.0
        pavg_min, pavg_max = 40.0, 80.0   # if infeasible, relax to [10, 90]
        beta_max = np.deg2rad(80.0)       # avoid Euler singularity
        # states ordering: [w(3), alpha,beta,gamma, v(3), p(3)]
        IDX_BETA = 4
        IDX_Z = 11

        # ---- cost weights (tune as needed) ----
        Q = np.diag([
            5, 5, 30,        # omega
            20, 40, 200,     # angles alpha,beta,gamma
            5, 50, 10,       # velocities
            80, 120, 120     # positions x,y,z
        ])
        R = np.diag([50.0, 50.0, 0.5, 0.2])     # input deviation
        Rd = np.diag([50.0, 50.0, 0.02, 0.02])   # input rate

        xref = self.x_ref.reshape(nx, 1)
        uref = self.u_ref.reshape(nu, 1)

        cost = 0

        for k in range(N):
            # dynamics (multiple shooting)
            opti.subject_to(X[:, k + 1] == f_d(X[:, k], U[:, k]))

            # input bounds
            opti.subject_to(opti.bounded(-dmax, U[0, k], dmax))             # delta1
            opti.subject_to(opti.bounded(-dmax, U[1, k], dmax))             # delta2
            opti.subject_to(opti.bounded(pavg_min, U[2, k], pavg_max))      # Pavg
            opti.subject_to(opti.bounded(-pdiff_max, U[3, k], pdiff_max))   # Pdiff

            # state bounds
            opti.subject_to(X[IDX_Z, k] >= 0.0)                             # z >= 0
            opti.subject_to(opti.bounded(-beta_max, X[IDX_BETA, k], beta_max))

            # stage cost
            e = X[:, k] - xref[:, 0]
            du = U[:, k] - uref[:, 0]
            cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([du.T, R, du])

            if k > 0:
                dU = U[:, k] - U[:, k - 1]
                cost += ca.mtimes([dU.T, Rd, dU])

        # terminal constraints too
        opti.subject_to(X[IDX_Z, N] >= 0.0)
        opti.subject_to(opti.bounded(-beta_max, X[IDX_BETA, N], beta_max))

        # terminal cost (simple, class-style; can be improved with LQR-P if you want)
        eN = X[:, N] - xref[:, 0]
        cost += ca.mtimes([eN.T, (10.0 * Q), eN])

        opti.minimize(cost)

        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {"sb": "yes", "print_level": 0, "max_iter": 200, "tol": 1e-3},
        }
        opti.solver("ipopt", opts)

        self.ocp = {"opti": opti, "X": X, "U": U, "X0": X0}

        # initial guesses
        opti.set_initial(X, np.tile(self.x_ref.reshape(-1, 1), (1, N + 1)))
        opti.set_initial(U, np.tile(self.u_ref.reshape(-1, 1), (1, N)))

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        opti = self.ocp["opti"]
        X = self.ocp["X"]
        U = self.ocp["U"]
        X0 = self.ocp["X0"]

        x0 = np.asarray(x0, dtype=float).reshape(12,)
        opti.set_value(X0, x0.reshape(12, 1))

        # warm-start: shift previous solution
        if self._X_last is not None and self._U_last is not None:
            Xg = np.hstack([self._X_last[:, 1:], self._X_last[:, [-1]]])
            Ug = np.hstack([self._U_last[:, 1:], self._U_last[:, [-1]]])
            opti.set_initial(X, Xg)
            opti.set_initial(U, Ug)

        try:
            sol = opti.solve()
            x_ol = np.array(sol.value(X), dtype=float)   # (12, N+1)
            u_ol = np.array(sol.value(U), dtype=float)   # (4, N)
            u0 = u_ol[:, 0].copy()

            self._X_last = x_ol
            self._U_last = u_ol

        except RuntimeError:
            # fallback: use trim input clipped to bounds
            u0 = self.u_ref.copy()
            u0[0] = np.clip(u0[0], -0.26, 0.26)
            u0[1] = np.clip(u0[1], -0.26, 0.26)
            u0[2] = np.clip(u0[2], 40.0, 80.0)
            u0[3] = np.clip(u0[3], -20.0, 20.0)

            x_ol = np.tile(x0.reshape(12, 1), (1, self.N + 1))
            u_ol = np.tile(u0.reshape(4, 1), (1, self.N))

        t_ol = t0 + np.arange(self.N + 1) * self.Ts
        return u0, x_ol, u_ol, t_ol
