import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron
from mpt4py.base import HData

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    PAVG_MIN = 40
    PAVG_MAX = 80



    def _setup_controller(self) -> None:

        A = self.A
        B = self.B
        nx, nu, N = self.nx, self.nu, self.N   # nx=1, nu=1

        # Q,R must match nx=1, nu=1
        Q = np.array([[50.0]])
        R = np.array([[1]])

        K_lqr, P, _ = dlqr(A, B, Q, R)

        # delta bounds for Pavg
        Pavg_s = float(self.us[0])
        du_min = self.PAVG_MIN - Pavg_s
        du_max = self.PAVG_MAX - Pavg_s

        vz_max = 15.0  # m/s loose bound for set computation

        # X 
        Hx = np.array([[1.0],
                       [-1.0]], dtype=float)
        hx = np.array([vz_max, vz_max], dtype=float)
        # U 
        Hu = np.array([[1.0],
                       [-1.0]], dtype=float)
        hu = np.array([du_max, -du_min], dtype=float)

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

        # compute maximal invariant set
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

        # terminal constraint and cost
        constraints += [Af @ (dx[:, N] - dxt_p) <= bf]
        cost += cp.quad_form(dx[:, N] - dxt_p, P)

        self._K_lqr = K_lqr
        self._P = P
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

        # reconstruct trajectories
        u_traj = self.us.reshape(-1, 1) + du_opt              # (1, N)
        x_traj = self.xs.reshape(-1, 1) + dx_opt              # (1, N+1)
        u0 = u_traj[:, 0].reshape(-1)                         # (1,)

        return u0, x_traj, u_traj