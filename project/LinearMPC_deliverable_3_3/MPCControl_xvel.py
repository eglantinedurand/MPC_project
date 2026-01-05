import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron
from mpt4py.base import HData

from .MPCControl_base import MPCControl_base


from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])
    DELTA2_MAX_DEG = 15.0
    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        A = self.A  # discrete subsystem (nx x nx)
        B = self.B  # discrete subsystem (nx x nu)
        nx, nu, N = self.nx, self.nu, self.N


        # -------- tuning (start here) --------
        # State order: [ωy, β, vx]
        Q = np.diag([500.0, 500.0, 100.0])
        R = np.diag([1000.0])
        # -------------------------------------

        # Terminal LQR cost
        # dlqr returns K such that u = -Kx
        K_lqr, P, _ = dlqr(A, B, Q, R)

        # ---------- constraints (absolute -> delta) ----------
        beta_max = np.deg2rad(10.0)
        beta_s = float(self.xs[1])

        delta2_max = np.deg2rad(float(self.DELTA2_MAX_DEG))
        delta2_s = float(self.us[0])

        # delta bounds
        dx_beta_min = -beta_max - beta_s
        dx_beta_max = +beta_max - beta_s
        du_min = -delta2_max - delta2_s
        du_max = +delta2_max - delta2_s
        print("delta2_s (trim) =", delta2_s)
        print("du bounds =", du_min, du_max)
        print("=> abs bounds =", delta2_s + du_min, delta2_s + du_max)


        omega_max = 5.0   #rad/s
        vx_max = 10.0  #m/s

        Hx = np.array([
            [ 1.0, 0.0, 0.0],   #  omega_y <= omega_max
            [-1.0, 0.0, 0.0],   # -omega_y <= omega_max
            [ 0.0, 1.0, 0.0],   #  beta    <= dx_beta_max
            [ 0.0,-1.0, 0.0],   # -beta    <= -dx_beta_min
            [ 0.0, 0.0, 1.0],   #  v_x     <= vx_max
            [ 0.0, 0.0,-1.0],   # -v_x     <= vx_max
        ], dtype=float)

        hx = np.array([
            omega_max,
            omega_max,
            dx_beta_max,
            -dx_beta_min,
            vx_max,
            vx_max
        ], dtype=float)

        X = Polyhedron(HData(Hx, hx))
        Hu = np.array([[ 1.0],
                       [-1.0]], dtype=float)
        hu = np.array([du_max, -du_min], dtype=float)
        U = Polyhedron(HData(Hu, hu))

        HK = -Hu @ K_lqr            # shape (2, nx)
        hK = hu
        XK = Polyhedron(HData(HK, hK))
        print([n for n in dir(X) if n in ["H","A","h","b"]])

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
        for _ in range(100):
            Ai = Xi.A
            hi = Xi.b
            PreXi = Polyhedron(HData(Ai @ Acl, hi))   # {x | Acl x ∈ Xi}
            Xnext = PreXi.intersect(X0)
            if _equal_poly(Xnext, Xi):
                Xi = Xnext
                break
            Xi = Xnext

        Xf = Xi
        self._Xf = Xf  
        print([n for n in dir(Xi) if n in ["H","h","A","b"]])

        Af, bf = Xf.A, Xf.b

        # ---------- build QP in delta variables ----------
        dx = cp.Variable((nx, N + 1))
        du = cp.Variable((nu, N))

        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)  # desired delta state (0 for stabilization)
        dut_p = cp.Parameter(nu)  # desired delta input (0 for stabilization)

        constraints = [dx[:, 0] == dx0_p]
        cost = 0

        for k in range(N):
            # dynamics
            constraints += [dx[:, k + 1] == A @ dx[:, k] + B @ du[:, k]]

            # beta constraint (delta)
            constraints += [dx[1, k] <= dx_beta_max]
            constraints += [dx[1, k] >= dx_beta_min]

            # input constraint (delta)
            constraints += [du[0, k] <= du_max]
            constraints += [du[0, k] >= du_min]

            # stage cost
            cost += cp.quad_form(dx[:, k] - dxt_p, Q) + cp.quad_form(du[:, k] - dut_p, R)

        # terminal beta bounds (keep constraints at terminal)
        constraints += [dx[1, N] <= dx_beta_max]
        constraints += [dx[1, N] >= dx_beta_min]

        # terminal constraints
        constraints += [Af @ (dx[:, N] - dxt_p) <= bf]

        # terminal cost
        cost += cp.quad_form(dx[:, N] - dxt_p, P)

        self._Q = Q
        self._R = R
        self._P = P
        self._K_lqr = K_lqr

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

        # delta initial / targets
        dx0 = x0 - self.xs
        dxt = x_target - self.xs
        dut = u_target - self.us

        self._dx0_p.value = dx0
        self._dxt_p.value = dxt
        self._dut_p.value = dut

        delta2_max = np.deg2rad(float(self.DELTA2_MAX_DEG))

        def saturate_u_abs(u_abs: np.ndarray) -> np.ndarray:
            # u_abs is subsystem absolute input (shape (1,))
            return np.clip(u_abs, -delta2_max, +delta2_max)

        # Solve
        self.ocp.solve(solver=cp.OSQP, warm_start=True)
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            du0 = (-self._K_lqr @ dx0).reshape(-1)
            u0 = self.us + du0                    # absolute
            u0 = saturate_u_abs(u0)               # enforce delta2 bounds

            return (
                u0,
                np.tile(x0.reshape(-1, 1), (1, self.N + 1)),
                np.tile(u0.reshape(-1, 1), (1, self.N)),
            )


        du_opt = np.array(self._du.value)
        dx_opt = np.array(self._dx.value)

        # Return ABSOLUTE trajectories for plotting/simulation consistency
        u_traj = self.us.reshape(-1, 1) + du_opt          # (1, N)
        u_traj = np.clip(u_traj, -delta2_max, +delta2_max)
        u0 = u_traj[:, 0]

        x_traj = self.xs.reshape(-1, 1) + dx_opt          # (3, N+1)
        u0 = u_traj[:, 0]



        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj