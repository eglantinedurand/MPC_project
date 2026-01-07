import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron
from mpt4py.base import HData

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    # full state ordering is [ω(3), φ(3), v(3), p(3)]
    # indices [8, 11] correspond to [v_z, z]
    x_ids: np.ndarray = np.array([8, 11])  # [vz, z]
    u_ids: np.ndarray = np.array([2])      # P_avg channel

    PAVG_MIN = 40.0
    PAVG_MAX = 80.0

    def _setup_controller(self) -> None:
        # Reduced + discretized by MPCControl_base
        A = np.array(self.A, dtype=float)  # (2,2)
        B = np.array(self.B, dtype=float)  # (2,1)
        nx, nu, N = self.nx, self.nu, self.N

        # -------------------- TUNING --------------------
        # State is DELTA [vz, z]
        Q = np.diag([79, 120])
        R = np.array([[0.24]])

        # Strong terminal soft penalty (helps anchoring at target without hard z>=3)
        wT = 2e5

        # Disturbance bounds (given)
        w_min, w_max = -15.0, 5.0

        # -------------------- Tube feedback K --------------------
        # dlqr uses u = -K_lqr x
        K_lqr, P, _ = dlqr(A, B, Q, R)
        K_tube = -K_lqr  # so Acl_err = A + B*K_tube = A - B*K_lqr

        # -------------------- Invariant error set E (box) via geometric series --------------------
        # e+ = (A + B K_tube)e + B w
        Acl_err = A + B @ K_tube

        w_abs = max(abs(w_min), abs(w_max))
        Bw_abs = np.abs(B.flatten()) * w_abs

        E_ub = np.zeros(nx)
        M = np.eye(nx)
        for _ in range(600):  # stable but close to 1 -> take enough terms
            E_ub += np.abs(M) @ Bw_abs
            M = Acl_err @ M
        E_lb = -E_ub

        self.E_lb = E_lb
        self.E_ub = E_ub

        # Polyhedron for plotting E
        I = np.eye(nx)
        H_E = np.vstack([I, -I])
        h_E = np.hstack([E_ub, -E_lb])
        self.E_poly = Polyhedron(HData(H_E, h_E))

        # -------------------- Tighten input bounds correctly --------------------
        us = float(self.us[0])
        du_min = self.PAVG_MIN - us
        du_max = self.PAVG_MAX - us

        # ke_min = min_{e in box} K_tube e
        # ke_max = max_{e in box} K_tube e
        k = K_tube.reshape(1, -1)  # (1,2)
        k_pos = np.maximum(k, 0.0)
        k_neg = np.minimum(k, 0.0)
        ke_min = float((k_pos @ E_lb + k_neg @ E_ub).reshape(-1)[0])
        ke_max = float((k_pos @ E_ub + k_neg @ E_lb).reshape(-1)[0])

        # CORRECT tube tightening:
        # v_min = du_min - max(Ke), v_max = du_max - min(Ke)
        v_min_tight = du_min - ke_max
        v_max_tight = du_max - ke_min

        self.v_min_tight = float(v_min_tight)
        self.v_max_tight = float(v_max_tight)

        # vertices of tightened input set in ABSOLUTE u (for deliverable printout)
        self.U_tight_vertices = np.array([us + v_min_tight, us + v_max_tight], dtype=float)

        # -------------------- Tighten z >= 0 (robust safety) --------------------
        # absolute: z >= 0
        # delta: z_s + Δz >= 0  -> Δz >= -z_s
        # robust: Δz_nom >= -z_s - min(e_z) = -z_s - E_lb[z]
        z_s = float(self.xs[1])
        z_margin = 0.02
        z_min_tight = -z_s - float(E_lb[1]) + z_margin
        self.z_min_tight = float(z_min_tight)

        # -------------------- Terminal set Xf (for deliverable + terminal constraint) --------------------
        # Keep it reasonable; main anchoring will come from strong terminal penalty + terminal constraint.
        vz_max = 3.0
        z_max = 3.0

        Hx = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=float,
        )
        hx = np.array([vz_max, vz_max, z_max, -z_min_tight], dtype=float)
        X = Polyhedron(HData(Hx, hx))

        Hu = np.array([[1.0], [-1.0]], dtype=float)
        hu = np.array([v_max_tight, -v_min_tight], dtype=float)
        U = Polyhedron(HData(Hu, hu))

        HK = -Hu @ K_lqr  # v = -K_lqr x
        XK = Polyhedron(HData(HK, hu))

        X0 = X.intersect(XK)

        self._X = X
        self._U = U
        self._XK = XK
        self._X0 = X0

        Acl_term = A - B @ K_lqr

        def _equal_poly(P1: Polyhedron, P2: Polyhedron) -> bool:
            try:
                return P1.is_subset(P2) and P2.is_subset(P1)
            except Exception:
                return False

        Xi = X0
        for _ in range(120):
            Ai, bi = Xi.A, Xi.b
            PreXi = Polyhedron(HData(Ai @ Acl_term, bi))
            Xnext = PreXi.intersect(X0)
            if _equal_poly(Xnext, Xi):
                Xi = Xnext
                break
            Xi = Xnext

        Xf = Xi
        self._Xf = Xf
        Af, bf = Xf.A, Xf.b

        # -------------------- OCP --------------------
        dx = cp.Variable((nx, N + 1))  # nominal delta state
        dv = cp.Variable((nu, N))      # nominal delta input v

        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)
        dut_p = cp.Parameter(nu)

        constraints = []
        cost = 0.0

        # Force nominal initial state to measured delta state (prevents “nominal shifting”)
        constraints += [dx[:, 0] == dx0_p]

        for kstep in range(N):
            constraints += [dx[:, kstep + 1] == A @ dx[:, kstep] + B @ dv[:, kstep]]

            # tightened input bounds on v
            constraints += [dv[0, kstep] >= v_min_tight, dv[0, kstep] <= v_max_tight]

            # robust tightened safety constraint z>=0
            constraints += [dx[1, kstep] >= z_min_tight]

            cost += cp.quad_form(dx[:, kstep] - dxt_p, Q) + cp.quad_form(dv[:, kstep] - dut_p, R)

        # terminal set + safety
        constraints += [Af @ (dx[:, N] - dxt_p) <= bf]
        constraints += [dx[1, N] >= z_min_tight]

        # terminal costs
        cost += cp.quad_form(dx[:, N] - dxt_p, P)
        cost += wT * cp.sum_squares(dx[:, N] - dxt_p)

        # Store
        self._K_lqr = K_lqr
        self._K_tube = K_tube
        self._Q = Q
        self._R = R
        self._P = P

        self._dx = dx
        self._dv = dv
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

        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        # print("\n=== MPC Z STEP DEBUG ===")
        # print("dx0 =", dx0, " (Δz0=", dx0[1], ")")
        # print("status =", self.ocp.status)
        # if self.ocp.status in ("optimal", "optimal_inaccurate"):
        #     print("dv0 =", float(self._dv.value[0,0]), " -> u_abs =", float(self.us[0] + self._dv.value[0,0]))
        # else:
        #     du_fb = float((self._K_tube @ dx0.reshape(self.nx, 1)).reshape(-1)[0])
        #     print("FALLBACK tube du =", du_fb, " -> u_abs =", float(np.clip(self.us[0] + du_fb, self.PAVG_MIN, self.PAVG_MAX)))

        # fallback: tube feedback only (keep within absolute bounds)
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            du0 = float((self._K_tube @ dx0.reshape(self.nx, 1)).reshape(-1)[0])
            u0_abs = float(np.clip(float(self.us[0]) + du0, self.PAVG_MIN, self.PAVG_MAX))
            u0 = np.array([u0_abs], dtype=float)

            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        dv_opt = np.array(self._dv.value, dtype=float)  # (1,N)
        dx_opt = np.array(self._dx.value, dtype=float)  # (2,N+1)

        # Since dx[:,0]==dx0, tube correction at k=0 is zero -> apply u = us + v0
        du0 = float(dv_opt[0, 0])
        u0_abs = float(self.us[0]) + du0
        u0_abs = float(np.clip(u0_abs, self.PAVG_MIN, self.PAVG_MAX))
        u0 = np.array([u0_abs], dtype=float)

        # For plotting (absolute)
        u_traj = self.us.reshape(-1, 1) + dv_opt
        x_traj = self.xs.reshape(-1, 1) + dx_opt
        u_traj = np.clip(u_traj, self.PAVG_MIN, self.PAVG_MAX)

        return u0, x_traj, u_traj

    # keep template API
    def setup_estimator(self):
        self.d_estimate = np.zeros((1,))
        self.d_gain = 0.0

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        self.d_estimate = self.d_estimate
