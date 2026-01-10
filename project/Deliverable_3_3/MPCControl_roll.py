import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron
from mpt4py.base import HData

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids = np.array([2, 5])   # omega_z, gamma
    u_ids = np.array([3])      # Pdiff

    PDIFF_MIN = -20.0
    PDIFF_MAX =  20.0

    def _setup_controller(self) -> None:
  
        A = self.A
        B = self.B
        nx, nu, N = self.nx, self.nu, self.N

        # Tuning
        Q = np.diag([100, 100])   # penalize gamma strongly
        R = np.array([[0.3]])  # penalize Pdiff usage

        K_lqr, P, _ = dlqr(A, B, Q, R)

        # Pdiff bounds in delta-coordinates
        Pdiff_s = float(self.us[0])
        du_min = self.PDIFF_MIN - Pdiff_s
        du_max = self.PDIFF_MAX - Pdiff_s

        omega_max = 10.0                 # rad/s loose
        gamma_max = np.deg2rad(180.0)    # loose but bounded for plotting

        Hx = np.array([
            [ 1, 0],
            [-1, 0],   # omega_z
            [ 0, 1],
            [ 0,-1],   # gamma
        ], float)

        hx = np.array([
            omega_max,
            omega_max,
            gamma_max,
            gamma_max
        ], float)

        Hu = np.array([
            [ 1.0],
            [-1.0]
        ], float)

        hu = np.array([
            du_max,
            -du_min
        ], float)

        # Build Polyhedra X and U
        X = Polyhedron(HData(Hx, hx))
        U = Polyhedron(HData(Hu, hu))

        # terminal law u = -Kx 
        HK = -Hu @ K_lqr
        XK = Polyhedron(HData(HK, hu))

        X0 = X.intersect(XK)
        self._X = X
        self._U = U
        self._XK = XK
        self._X0 = X0

        Acl = A - B @ K_lqr

        def _equal_poly(P1: Polyhedron, P2: Polyhedron) -> bool:
            try:
                return P1.is_subset(P2) and P2.is_subset(P1)
            except Exception:
                return False

        Xi = X0

        # Compute maximal invariant set
        for _ in range(100):
            Ai, hi = Xi.A, Xi.b
            PreXi = Polyhedron(HData(Ai @ Acl, hi))
            Xnext = PreXi.intersect(X0)
            if _equal_poly(Xnext, Xi):
                Xi = Xnext
                break
            Xi = Xnext

        Xf = Xi
        self._Xf = Xf
        Af, bf = Xf.A, Xf.b

        dx = cp.Variable((nx, N + 1))
        du = cp.Variable((nu, N))

        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)
        dut_p = cp.Parameter(nu)

        constraints = [dx[:, 0] == dx0_p]
        cost = 0

        for k in range(N):
            constraints += [
                dx[:, k+1] == A @ dx[:, k] + B @ du[:, k]
            ]
            constraints += [
                du[0, k] >= du_min,
                du[0, k] <= du_max
            ]
            cost += (
                cp.quad_form(dx[:, k] - dxt_p, Q)
                + cp.quad_form(du[:, k] - dut_p, R)
            )

        # terminal cost and constraint
        cost += cp.quad_form(dx[:, N] - dxt_p, P)
        constraints += [Af @ (dx[:, N] - dxt_p) <= bf]

        self._Q, self._R, self._P, self._K_lqr = Q, R, P, K_lqr
        self._dx, self._du = dx, du
        self._dx0_p, self._dxt_p, self._dut_p = dx0_p, dxt_p, dut_p

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self,
        x0: np.ndarray,
        x_target: np.ndarray = None,
        u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

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

        # solve OCP
        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        # fallback to LQR if problem is infeasible
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            du0 = (-self._K_lqr @ dx0).reshape(-1)
            u0 = self.us + du0
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        du_opt = np.array(self._du.value)
        dx_opt = np.array(self._dx.value)

        # reconstruct trajectories
        u_traj = self.us.reshape(-1, 1) + du_opt
        x_traj = self.xs.reshape(-1, 1) + dx_opt
        u0 = u_traj[:, 0]
        
        return u0, x_traj, u_traj
