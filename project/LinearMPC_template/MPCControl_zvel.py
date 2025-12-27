import numpy as np
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    PAVG_MIN = 40
    PAVG_MAX = 80



    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        A = self.A
        B = self.B
        nx, nu, N = self.nx, self.nu, self.N   # nx=1, nu=1

        # Q,R must match nx=1, nu=1
        Q = np.array([[80.0]])
        R = np.array([[0.5]])

        K_lqr, P, _ = dlqr(A, B, Q, R)

        # delta bounds for Pavg (absolute bounds + hover band)
        Pavg_s = float(self.us[0])
        umin_abs = max(self.PAVG_MIN, Pavg_s)
        umax_abs = min(self.PAVG_MAX, Pavg_s)

        du_min = float(umin_abs - Pavg_s)
        du_max = float(umax_abs - Pavg_s)

        # decision variables
        dx = cp.Variable((nx, N + 1))
        du = cp.Variable((nu, N))

        # parameters
        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)
        dut_p = cp.Parameter(nu)

        constraints = [dx[:, 0] == dx0_p]
        cost = 0

        for k in range(N):
            constraints += [dx[:, k + 1] == A @ dx[:, k] + B @ du[:, k]]
            constraints += [du[0, k] <= du_max, du[0, k] >= du_min]
            cost += cp.quad_form(dx[:, k] - dxt_p, Q) + cp.quad_form(du[:, k] - dut_p, R)

        cost += cp.quad_form(dx[:, N] - dxt_p, P)

        self._K_lqr = K_lqr
        self._P = P
        self._dx = dx
        self._du = du
        self._dx0_p = dx0_p
        self._dxt_p = dxt_p
        self._dut_p = dut_p

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        if x_target is None:
            x_target = self.xs
        if u_target is None:
            u_target = self.us

        dx0 = x0 - self.xs
        dxt = x_target - self.xs
        dut = u_target - self.us

        self._dx0_p.value = dx0
        self._dxt_p.value = dxt
        self._dut_p.value = dut

        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        # fallback
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            du0 = (-self._K_lqr @ dx0).reshape(-1)            # (1,)
            u0 = (self.us + du0).reshape(-1)                  # (1,)
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        du_opt = np.array(self._du.value)                     # (1, N)
        dx_opt = np.array(self._dx.value)                     # (1, N+1)

        u_traj = self.us.reshape(-1, 1) + du_opt              # (1, N)
        x_traj = self.xs.reshape(-1, 1) + dx_opt              # (1, N+1)
        u0 = u_traj[:, 0].reshape(-1)                         # (1,)

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj