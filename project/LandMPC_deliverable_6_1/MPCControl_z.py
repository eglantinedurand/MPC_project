import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron
from mpt4py.base import HData

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    # full state ordering is [ω(3), φ(3), v(3), p(3)]
    # so indices [8, 11] correspond to [v_z, z]
    x_ids: np.ndarray = np.array([8, 11])  # [vz, z]
    u_ids: np.ndarray = np.array([2])      # P_avg channel

    PAVG_MIN = 40
    PAVG_MAX = 80

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        A = self.A
        B = self.B
        nx, nu, N = self.nx, self.nu, self.N  # nx=2, nu=1 for sys_z

        # -------------------- MPC weights (state is [vz, z]) --------------------
        Q = np.diag([20.0, 40.0])
        R = np.array([[8]])

        # Soft “don’t go below target” (undershoot) penalty weight
        self.w_under = 2e4

        # Integral action (offset-free) weights
        self.w_int = 50.0
        self.w_int_T = 200.0

        # LQR (dlqr uses u = -K_lqr x)
        K_lqr, P, _ = dlqr(A, B, Q, R)

        # Tube feedback gain in u = v + K_tube e convention
        # so Acl_err = A + B K_tube = A - B K_lqr
        K_tube = -K_lqr

        # -------------------- disturbance bounds (given) --------------------
        w_min, w_max = -15.0, 5.0

        # -------------------- invariant error set E: box outer-approx via geometric series --------------------
        Acl_err = A + B @ K_tube

        w_abs = max(abs(w_min), abs(w_max))  # 15
        Bw_abs = np.abs(B.flatten()) * w_abs

        E_ub = np.zeros(nx)
        M = np.eye(nx)
        for _ in range(500):
            E_ub += np.abs(M) @ Bw_abs
            M = Acl_err @ M
        E_lb = -E_ub

        self.E_lb = E_lb
        self.E_ub = E_ub

        # Polyhedron for plotting E (HData style)
        I = np.eye(nx)
        H_E = np.vstack([I, -I])
        h_E = np.hstack([E_ub, -E_lb])
        self.E_poly = Polyhedron(HData(H_E, h_E))

        # -------------------- input constraints (absolute) -> delta constraints for Δu --------------------
        Pavg_s = float(self.us[0])
        du_min = self.PAVG_MIN - Pavg_s
        du_max = self.PAVG_MAX - Pavg_s

        # -------------------- tighten input for nominal v: v ∈ ΔU ⊖ (K_tube E) --------------------
        k = K_tube.reshape(-1)  # (2,)
        ke_min = 0.0
        ke_max = 0.0
        for j in range(nx):
            if k[j] >= 0:
                ke_min += k[j] * E_lb[j]
                ke_max += k[j] * E_ub[j]
            else:
                ke_min += k[j] * E_ub[j]
                ke_max += k[j] * E_lb[j]

        v_min_tight = du_min - ke_max
        v_max_tight = du_max - ke_min
        self.v_min_tight = float(v_min_tight)
        self.v_max_tight = float(v_max_tight)

        # vertices of tightened INPUT set (absolute u endpoints)
        self.U_tight_vertices = np.array([
            float(self.us + v_min_tight),
            float(self.us + v_max_tight),
        ])

        # -------------------- tighten state constraint: z >= 0 --------------------
        xs_z = float(self.xs[1])
        z_margin = 0.02  # 2 cm numerical buffer
        z_min_tight = -xs_z - float(E_lb[1]) + z_margin
        self.z_min_tight = float(z_min_tight)

        # -------------------- terminal set Xf (polyhedron) --------------------
        vz_max = 5.0
        z_max = 12.0

        Hx = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ], dtype=float)
        hx = np.array([vz_max, vz_max, z_max, -z_min_tight], dtype=float)
        X = Polyhedron(HData(Hx, hx))

        Hu = np.array([[1.0], [-1.0]], dtype=float)
        hu = np.array([v_max_tight, -v_min_tight], dtype=float)
        U = Polyhedron(HData(Hu, hu))

        HK = -Hu @ K_lqr
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
        for _ in range(100):
            Ai, hi = Xi.A, Xi.b
            PreXi = Polyhedron(HData(Ai @ Acl_term, hi))
            Xnext = PreXi.intersect(X0)
            if _equal_poly(Xnext, Xi):
                Xi = Xnext
                break
            Xi = Xnext

        Xf = Xi
        self._Xf = Xf
        Af, bf = Xf.A, Xf.b

        # -------------------- build tube MPC OCP --------------------
        dx = cp.Variable((nx, N + 1))     # nominal delta state [vz,z]
        dv = cp.Variable((nu, N))         # nominal delta input

        # integral action state (scalar): integrate (z - z_ref)
        i_var = cp.Variable((1, N + 1))

        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)
        dut_p = cp.Parameter(nu)
        i0_p = cp.Parameter(1)            # stored integrator value

        constraints = []
        cost = 0

        # tube initial condition: dx_nom0 ∈ dx0 - E
        constraints += [
            dx[:, 0] >= dx0_p - E_ub,
            dx[:, 0] <= dx0_p - E_lb,
        ]

        # integrator initial condition
        constraints += [i_var[:, 0] == i0_p]

        for kstep in range(N):
            constraints += [dx[:, kstep + 1] == A @ dx[:, kstep] + B @ dv[:, kstep]]

            # integral dynamics: i_{k+1} = i_k + Ts*(z - z_ref)
            constraints += [
                i_var[:, kstep + 1]
                == i_var[:, kstep] + self.Ts * (dx[1, kstep] - dxt_p[1])
            ]

            # tightened input bounds for dv
            constraints += [dv[0, kstep] <= v_max_tight, dv[0, kstep] >= v_min_tight]

            # tightened z>=0 constraint for nominal
            constraints += [dx[1, kstep] >= z_min_tight]

            # standard tracking + input effort
            cost += cp.quad_form(dx[:, kstep] - dxt_p, Q) + cp.quad_form(dv[:, kstep] - dut_p, R)

            # (1) soft “no undershoot” penalty: penalize z < z_ref
            undershoot = cp.pos(dxt_p[1] - dx[1, kstep])
            cost += self.w_under * cp.square(undershoot)

            # (2) integral action penalty
            cost += self.w_int * cp.square(i_var[0, kstep])

        # terminal constraint: dx_N - dxt ∈ Xf
        constraints += [Af @ (dx[:, N] - dxt_p) <= bf]
        constraints += [dx[1, N] >= z_min_tight]

        # terminal costs
        cost += cp.quad_form(dx[:, N] - dxt_p, P)
        cost += self.w_int_T * cp.square(i_var[0, N])

        # store for get_u()
        self._K_lqr = K_lqr
        self._K_tube = K_tube
        self._P = P
        self._dx = dx
        self._dv = dv
        self._i = i_var
        self._dx0_p = dx0_p
        self._dxt_p = dxt_p
        self._dut_p = dut_p
        self._i0_p = i0_p

        # integrator memory (persistent across calls)
        self.i_state = np.zeros((1,), dtype=float)
        self.i_clip = 50.0  # anti-windup clamp (tunable)

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

        # set parameters
        self._dx0_p.value = dx0
        self._dxt_p.value = dxt
        self._dut_p.value = dut
        self._i0_p.value = self.i_state

        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        # update integrator state using measured error (discrete integral)
        self.i_state = self.i_state + self.Ts * np.array([dx0[1] - dxt[1]])
        self.i_state = np.clip(self.i_state, -self.i_clip, self.i_clip)

        # fallback: pure tube feedback with v = 0 (plus no integral injection to u here)
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            du0 = (self._K_tube @ dx0).reshape(-1)
            u0 = (self.us + du0).reshape(-1)
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        dv_opt = np.array(self._dv.value)   # (1, N)
        dx_opt = np.array(self._dx.value)   # (2, N+1)

        # tube law: Δu = v0 + K_tube (Δx0 - dx_nom0)
        dx_nom0 = dx_opt[:, 0]
        e0 = dx0 - dx_nom0
        du0 = (dv_opt[:, 0] + (self._K_tube @ e0)).reshape(-1)
        u0 = (self.us + du0).reshape(-1)

        # return predicted trajectories in absolute coordinates (for plotting)
        u_traj = self.us.reshape(-1, 1) + dv_opt
        x_traj = self.xs.reshape(-1, 1) + dx_opt

        # YOUR CODE HERE
        #################################################
        return u0, x_traj, u_traj

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        self.d_estimate = np.zeros((1,))
        self.d_gain = 0.0
        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        self.d_estimate = self.d_estimate
        # YOUR CODE HERE
        ##################################################
