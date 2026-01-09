import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron
from mpt4py.base import HData

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    """
    Option 3 (course-aligned): estimated disturbance in nominal prediction + residual tube MPC.

    Simulator for sys_z (rocket.py):
        dx_{k+1} = A dx_k + B (du_k + w_k)

    Controller:
    - estimate w_hat online
    - nominal MPC predicts: dx_nom+ = A dx_nom + B (dv + w_hat)
    - apply: du = dv0 + sat_tube( K_tube (dx_meas - dx_nom) )   (IMPORTANT)
      where tube correction is saturated so the final du respects actuator bounds
      WITHOUT breaking the tube assumptions.
    """

    x_ids: np.ndarray = np.array([8, 11])  # [vz, z]
    u_ids: np.ndarray = np.array([2])      # P_avg channel

    PAVG_MIN = 40.0
    PAVG_MAX = 80.0

    def _setup_controller(self) -> None:
        A = np.array(self.A, dtype=float)  # (2,2)
        B = np.array(self.B, dtype=float)  # (2,1)
        nx, nu = self.nx, self.nu

        # ----- Horizon handling (H may be seconds or steps) -----
        Ts = float(getattr(self, "Ts", 0.05))
        H_in = getattr(self, "H", None)
        if H_in is None:
            N = int(self.N)
        else:
            if float(H_in) <= 20.0:  # treat as seconds
                N = int(max(2, round(float(H_in) / Ts)))
            else:  # treat as steps
                N = int(max(2, round(float(H_in))))
        self.N = N

        # ----- Costs -----
        Q = np.diag([200.0, 200.0])  # [vz, z]
        R = np.array([[0.1]])

        # ----- Disturbance bounds (given) -----
        self.w_min, self.w_max = -15.0, 5.0

        # Robustness against residual w_res = w - w_hat.
        # Worst-case if w_hat clipped to [w_min,w_max]:
        #   w_res in [w_min-w_max, w_max-w_min] = [-20, +20]
        EPS_W = 20.0
        self.wres_min, self.wres_max = -EPS_W, +EPS_W

        # ----- Tube gain (dlqr returns u=-Kx) -----
        K_lqr, P, _ = dlqr(A, B, Q, R)
        K_tube = -K_lqr
        Acl_err = A + B @ K_tube

        self._K_lqr = K_lqr
        self._K_tube = K_tube
        self._Q = Q
        self._R = R
        self._P = P

        # ----- Invariant error set E for e+ = (A+BK)e + B w_res -----
        E_lb = np.zeros(nx, dtype=float)
        E_ub = np.zeros(nx, dtype=float)

        A_pow = np.eye(nx)
        for _ in range(1200):
            g = (A_pow @ B).reshape(nx)
            E_ub += np.maximum(g, 0.0) * self.wres_max + np.minimum(g, 0.0) * self.wres_min
            E_lb += np.minimum(g, 0.0) * self.wres_max + np.maximum(g, 0.0) * self.wres_min
            A_pow = Acl_err @ A_pow

        self.E_lb, self.E_ub = E_lb, E_ub

        I = np.eye(nx)
        H_E = np.vstack([I, -I])
        h_E = np.hstack([E_ub, -E_lb])
        self.E_poly = Polyhedron(HData(H_E, h_E))

        # ----- Actuator bounds in delta form -----
        self.us0 = float(self.us[0])
        self.du_min = self.PAVG_MIN - self.us0
        self.du_max = self.PAVG_MAX - self.us0

        # ----- Tightened bounds on dv so dv + K e stays within du bounds -----
        k = K_tube.reshape(1, -1)
        k_pos = np.maximum(k, 0.0)
        k_neg = np.minimum(k, 0.0)

        ke_min = float((k_pos @ E_lb + k_neg @ E_ub).reshape(-1)[0])
        ke_max = float((k_pos @ E_ub + k_neg @ E_lb).reshape(-1)[0])

        dv_min_tight = self.du_min - ke_max
        dv_max_tight = self.du_max - ke_min

        self.dv_min_tight = float(dv_min_tight)
        self.dv_max_tight = float(dv_max_tight)
        self.U_tight_vertices = np.array([self.us0 + dv_min_tight, self.us0 + dv_max_tight], dtype=float)

        # ----- Tightened robust ground constraint on nominal -----
        z_s = float(self.xs[1])
        z_margin = 0.5
        z_min_tight = -z_s - float(E_lb[1]) + z_margin
        self.z_min_tight = float(z_min_tight)

        # ----- Terminal set Xf (loose box + maximal invariant subset) -----
        vz_max = 12.0
        z_max = 30.0

        Hx = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        hx = np.array([vz_max, vz_max, z_max, -z_min_tight], dtype=float)
        X = Polyhedron(HData(Hx, hx))

        Hu = np.array([[1.0], [-1.0]], dtype=float)
        hu = np.array([dv_max_tight, -dv_min_tight], dtype=float)
        U = Polyhedron(HData(Hu, hu))

        HK = -Hu @ K_lqr
        XK = Polyhedron(HData(HK, hu))
        X0 = X.intersect(XK)

        Acl_term = A - B @ K_lqr

        def _equal_poly(P1: Polyhedron, P2: Polyhedron) -> bool:
            try:
                return P1.is_subset(P2) and P2.is_subset(P1)
            except Exception:
                return False

        Xi = X0
        for _ in range(250):
            Ai, bi = Xi.A, Xi.b
            PreXi = Polyhedron(HData(Ai @ Acl_term, bi))
            Xnext = PreXi.intersect(X0)
            if _equal_poly(Xnext, Xi):
                Xi = Xnext
                break
            Xi = Xnext

        Xf = Xi
        self._X = X
        self._U = U
        self._Xf = Xf
        Af, bf = Xf.A, Xf.b

        # ===================== Nominal OCP (with w_hat in dynamics) =====================
        dx = cp.Variable((nx, N + 1))
        dv = cp.Variable((nu, N))

        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)
        dut_p = cp.Parameter(nu)
        w_hat_p = cp.Parameter(1)

        constraints = [dx[:, 0] == dx0_p]
        cost = 0.0

        for kstep in range(N):
            # nominal prediction includes estimated disturbance
            constraints += [dx[:, kstep + 1] == A @ dx[:, kstep] + B @ (dv[:, kstep] + w_hat_p)]

            # tightened actuator constraint on dv
            constraints += [dv[0, kstep] >= dv_min_tight, dv[0, kstep] <= dv_max_tight]

            # robust ground on nominal
            constraints += [dx[1, kstep] >= z_min_tight]

            cost += cp.quad_form(dx[:, kstep] - dxt_p, Q)
            cost += cp.quad_form(dv[:, kstep] - dut_p, R)

        constraints += [Af @ (dx[:, N] - dxt_p) <= bf]
        constraints += [dx[1, N] >= z_min_tight]
        cost += cp.quad_form(dx[:, N] - dxt_p, P)

        self._dx = dx
        self._dv = dv
        self._dx0_p = dx0_p
        self._dxt_p = dxt_p
        self._dut_p = dut_p
        self._w_hat_p = w_hat_p

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # ----- Tube MPC nominal state -----
        self.dx_bar = np.zeros(nx, dtype=float)
        self._dxbar_initialized = False

        # ----- Disturbance estimator memory -----
        self.w_hat = 0.0
        self.alpha_w = 0.7
        self._x_prev = None
        self._u_prev = None

        # Debug (optional)
        print("N:", self.N,
              "dv_tight:", self.dv_min_tight, self.dv_max_tight,
              "du_bounds:", self.du_min, self.du_max,
              "z_min_tight:", self.z_min_tight)

    def _update_w_estimator(self, x_now: np.ndarray) -> None:
        """
        One-step residual estimator:
            dx_now = A dx_prev + B (du_prev + w_prev)
        => w_hat from projection on B.
        """
        if self._x_prev is None or self._u_prev is None:
            return

        dx_prev = self._x_prev - self.xs
        dx_now = x_now - self.xs
        du_prev = float(self._u_prev - self.us0)

        resid = dx_now - (self.A @ dx_prev + self.B.flatten() * du_prev)

        b = self.B.flatten()
        w_raw = float((b @ resid) / (b @ b + 1e-9))
        w_raw = float(np.clip(w_raw, self.w_min, self.w_max))

        self.w_hat = (1.0 - self.alpha_w) * self.w_hat + self.alpha_w * w_raw

    def _apply_with_tube_saturation(self, dv0: float, tube_corr: float) -> tuple[np.ndarray, float]:
        """
        Compute applied input with tube correction saturation:
            du = dv0 + sat(tube_corr)   such that du in [du_min, du_max]
        This preserves the tube structure (no post-clipping of u breaking assumptions).
        """
        # Saturate tube correction (not u) to keep du feasible
        tube_corr_sat = float(np.clip(tube_corr, self.du_min - dv0, self.du_max - dv0))
        du_applied = dv0 + tube_corr_sat

        u0_abs = self.us0 + du_applied
        # redundant safety clip
        u0_abs = float(np.clip(u0_abs, self.PAVG_MIN, self.PAVG_MAX))
        return np.array([u0_abs], dtype=float), u0_abs

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):
        if x_target is None:
            x_target = self.xs
        if u_target is None:
            u_target = self.us

        # Update disturbance estimate from previous step
        self._update_w_estimator(x0)

        dx0 = x0 - self.xs
        dxt = x_target - self.xs
        dut = u_target - self.us  # usually 0

        # init nominal state
        if not self._dxbar_initialized:
            self.dx_bar = dx0.copy().astype(float)
            self._dxbar_initialized = True

        # Solve OCP from nominal dx_bar
        self._dx0_p.value = self.dx_bar
        self._dxt_p.value = dxt
        self._dut_p.value = dut
        self._w_hat_p.value = np.array([self.w_hat], dtype=float)

        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        # tube correction from current mismatch
        e0 = (dx0 - self.dx_bar).reshape(self.nx, 1)
        tube_corr = float((self._K_tube @ e0).reshape(-1)[0])

        # fallback
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            # feedback around target (still tube gain)
            e_track = (dx0 - dxt).reshape(self.nx, 1)
            dv0 = float((self._K_tube @ e_track).reshape(-1)[0])

            u0, u0_abs = self._apply_with_tube_saturation(dv0, 0.0)

            # store for estimator
            self._x_prev = x0.copy()
            self._u_prev = float(u0_abs)

            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        # extract optimal sequences
        dv_opt = np.array(self._dv.value, dtype=float)  # (1,N)
        dx_opt = np.array(self._dx.value, dtype=float)  # (2,N+1)
        dv0 = float(dv_opt[0, 0])

        # Apply u with saturated tube correction
        u0, u0_abs = self._apply_with_tube_saturation(dv0, tube_corr)

        # propagate nominal state with nominal dynamics (uses w_hat)
        self.dx_bar = (self.A @ self.dx_bar + self.B.flatten() * (dv0 + float(self.w_hat))).astype(float)

        # store for estimator
        self._x_prev = x0.copy()
        self._u_prev = float(u0_abs)

        # plotting trajectories (nominal)
        x_traj = self.xs.reshape(-1, 1) + dx_opt
        u_traj = self.us.reshape(-1, 1) + dv_opt
        u_traj = np.clip(u_traj, self.PAVG_MIN, self.PAVG_MAX)

        print("z=", x0[1], "vz=", x0[0], "u=", u0_abs, "w_hat=", self.w_hat, "dv0=", dv0, "tube=", tube_corr)


        return u0, x_traj, u_traj
