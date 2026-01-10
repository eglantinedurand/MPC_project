import numpy as np
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    # [ωz, γ] and input Pdiff
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        A = np.array(self.A, dtype=float)
        B = np.array(self.B, dtype=float)

        nx, nu = self.nx, self.nu
        N = int(self.N)

        # ---- differential throttle bounds (typical in project) ----
        # In Part 1: Pdiff may go up to ±20
        PDIFF_MAX = 20.0
        du_min = -PDIFF_MAX - float(self.us[0])
        du_max = +PDIFF_MAX - float(self.us[0])

        # state order: [ωz, γ]
        Q = np.diag([10.0, 100.0])
        R = np.array([[0.1]])

        K_lqr, _, _ = dlqr(A, B, Q, R)
        self._K_lqr = K_lqr

        dx = cp.Variable((nx, N + 1))
        du = cp.Variable((nu, N))

        # (optional) soft bounds on γ to avoid crazy spins in nonlinear sim
        # keep wide so it almost never binds, but preserves feasibility if it does
        gamma_soft_max = np.deg2rad(120.0)
        s_gamma = cp.Variable((N + 1,), nonneg=True)

        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)
        dut_p = cp.Parameter(nu)

        constraints = [dx[:, 0] == dx0_p]
        cost = 0.0
        SLACK_W = 1e4

        for k in range(N):
            constraints += [dx[:, k + 1] == A @ dx[:, k] + B @ du[:, k]]
            constraints += [du[0, k] >= du_min, du[0, k] <= du_max]

            # soft |γ| bound (γ is index 1)
            constraints += [dx[1, k] <= gamma_soft_max + s_gamma[k]]
            constraints += [-dx[1, k] <= gamma_soft_max + s_gamma[k]]

            cost += cp.quad_form(dx[:, k] - dxt_p, Q)
            cost += cp.quad_form(du[:, k] - dut_p, R)
            cost += SLACK_W * cp.square(s_gamma[k])

        constraints += [dx[1, N] <= gamma_soft_max + s_gamma[N]]
        constraints += [-dx[1, N] <= gamma_soft_max + s_gamma[N]]
        cost += cp.quad_form(dx[:, N] - dxt_p, Q)
        cost += SLACK_W * cp.square(s_gamma[N])

        self._dx = dx
        self._du = du
        self._dx0_p = dx0_p
        self._dxt_p = dxt_p
        self._dut_p = dut_p

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if x_target is None:
            x_target = self.xs
        if u_target is None:
            u_target = self.us

        dx0 = np.array(x0 - self.xs, dtype=float).reshape(-1)
        dxt = np.array(x_target - self.xs, dtype=float).reshape(-1)
        dut = np.array(u_target - self.us, dtype=float).reshape(-1)

        self._dx0_p.value = dx0
        self._dxt_p.value = dxt
        self._dut_p.value = dut

        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            e = (dx0 - dxt).reshape(-1, 1)
            du0 = float((-self._K_lqr @ e).reshape(-1)[0])
            u0 = np.array([float(self.us[0] + du0)], dtype=float)

            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        du_opt = np.array(self._du.value, dtype=float)
        dx_opt = np.array(self._dx.value, dtype=float)

        du0 = float(du_opt[0, 0])
        u0 = np.array([float(self.us[0] + du0)], dtype=float)

        x_traj = self.xs.reshape(-1, 1) + dx_opt
        u_traj = self.us.reshape(-1, 1) + du_opt

        return u0, x_traj, u_traj
