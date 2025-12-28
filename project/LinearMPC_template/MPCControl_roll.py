
import numpy as np
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids = np.array([0, 5])   # omega_x, gamma
    u_ids = np.array([3])      # Pdiff


    PDIFF_MIN = -20.0
    PDIFF_MAX =  20.0

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        A = self.A
        B = self.B
        nx, nu, N = self.nx, self.nu, self.N

        # Tuning (example)
        Q = np.diag([1, 500])   # penalize gamma strongly
        R = np.array([[100.0]])        # penalize Pdiff usage

        K_lqr, P, _ = dlqr(A, B, Q, R)

        # Pdiff bounds in delta-coordinates
        Pdiff_s = float(self.us[0])
        du_min = self.PDIFF_MIN - Pdiff_s
        du_max = self.PDIFF_MAX - Pdiff_s

        dx = cp.Variable((nx, N + 1))
        du = cp.Variable((nu, N))
        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)
        dut_p = cp.Parameter(nu)

        constraints = [dx[:, 0] == dx0_p]
        cost = 0

        for k in range(N):
            constraints += [dx[:, k+1] == A @ dx[:, k] + B @ du[:, k]]
            constraints += [du[0, k] >= du_min, du[0, k] <= du_max]
            cost += cp.quad_form(dx[:, k] - dxt_p, Q) + cp.quad_form(du[:, k] - dut_p, R)

        cost += cp.quad_form(dx[:, N] - dxt_p, P)

        self._Q, self._R, self._P, self._K_lqr = Q, R, P, K_lqr
        self._dx, self._du = dx, du
        self._dx0_p, self._dxt_p, self._dut_p = dx0_p, dxt_p, dut_p

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)


        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        if x_target is None: x_target = self.xs
        if u_target is None: u_target = self.us

        dx0 = x0 - self.xs
        dxt = x_target - self.xs
        dut = u_target - self.us

        self._dx0_p.value = dx0
        self._dxt_p.value = dxt
        self._dut_p.value = dut

        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            du0 = (-self._K_lqr @ dx0).reshape(-1)
            u0 = self.us + du0
            x_traj = np.tile(x0.reshape(-1,1), (1, self.N+1))
            u_traj = np.tile(u0.reshape(-1,1), (1, self.N))
            return u0, x_traj, u_traj

        du_opt = np.array(self._du.value)
        dx_opt = np.array(self._dx.value)

        u_traj = self.us.reshape(-1,1) + du_opt
        x_traj = self.xs.reshape(-1,1) + dx_opt
        u0 = u_traj[:, 0]
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
