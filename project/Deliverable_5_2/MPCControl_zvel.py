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
        Q = np.array([[30]])
        R = np.array([[1]])

        K_lqr, P, _ = dlqr(A, B, Q, R)

        # delta bounds for Pavg 
        Pavg_s = float(self.us[0])
        du_min = self.PAVG_MIN - Pavg_s
        du_max = self.PAVG_MAX - Pavg_s

        vz_max = 15.0  # m/s loose bound for set computation



        # decision variables
        dx = cp.Variable((nx, N + 1))
        du = cp.Variable((nu, N))

        # parameters
        dx0_p = cp.Parameter(nx)
        dxt_p = cp.Parameter(nx)
        dut_p = cp.Parameter(nu)
        d_p   = cp.Parameter(1) 

        constraints = [dx[:, 0] == dx0_p]
        cost = 0

        for k in range(N):
            constraints += [dx[:, k + 1] == A @ dx[:, k] + B @ du[:, k] + B @ d_p]
            constraints += [du[0, k] <= du_max, du[0, k] >= du_min]
            cost += cp.quad_form(dx[:, k] - dxt_p, Q) + cp.quad_form(du[:, k] - dut_p, R)


        cost += cp.quad_form(dx[:, N] - dxt_p, P)

        # store for use in get_u
        self._K_lqr = K_lqr
        self._P = P
        self._dx = dx
        self._du = du
        self._dx0_p = dx0_p
        self._dxt_p = dxt_p
        self._dut_p = dut_p
        self._d_p = d_p
        self._du_min = du_min
        self._du_max = du_max


        # estimator variables
        self._est_initialized = False
        self._xhat = np.zeros((1, 1))
        self._dhat = np.zeros((1, 1))
        self._du_prev = np.zeros((1, 1))

        self._Lx = np.array([[0.6]])
        self._Ld = np.array([[2]])


        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
   

    def get_u(
    self,
    x0: np.ndarray,
    x_target: np.ndarray = None,
    u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # defaults 
        if x_target is None:
            x_target = self.xs
        if u_target is None:
            u_target = self.us

        # measurement (treat x0 as y_k) 
        y = np.array(x0, dtype=float).reshape(1, 1)              # measured vz
        xs_col = np.array(self.xs, dtype=float).reshape(1, 1)    # equilibrium vz
        us_col = np.array(self.us, dtype=float).reshape(1, 1)    # equilibrium Pavg

        # delta measurement
        y_delta = y - xs_col                                    

        # initialize estimator 
        if self._est_initialized ==False:
            self._xhat = y_delta.copy()                          # (1,1) delta state estimate
            self._dhat = np.zeros((1, 1))                        # (1,1) disturbance estimate
            self._du_prev = np.zeros((1, 1))                     # (1,1) previous delta input
            self._est_initialized = True

        else:
            # observer update
            A = np.array(self.A, dtype=float)                    # (1,1)
            B = np.array(self.B, dtype=float)                    # (1,1)

            # prediction
            xhat_pred = A @ self._xhat + B @ self._du_prev + B @ self._dhat   # (1,1)

            # gains
            e = y_delta - xhat_pred                              # (1,1)
            Lx = np.array(getattr(self, "_Lx", [[0.6]]), dtype=float).reshape(1, 1)
            Ld = np.array(getattr(self, "_Ld", [[0.05]]), dtype=float).reshape(1, 1)

            # correction
            self._xhat = xhat_pred + Lx @ e                      # (1,1)
            self._dhat = self._dhat + Ld @ e                     # (1,1)

        # MPC parameters
        dx0 = self._xhat.reshape(1,)                             # (1,)
        dhat = self._dhat.reshape(1,)                            # (1,)

        dxt = np.array(x_target - self.xs, dtype=float).reshape(1,)   # (1,)
        dut = np.array([-dhat.item()], dtype=float).reshape(1,)
                                                         

        self._dx0_p.value = dx0
        self._dxt_p.value = dxt
        self._dut_p.value = dut
        self._d_p.value = dhat

        # solve QP 
        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        # fallback 
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            # LQR on estimated delta state
            du0 = float((-self._K_lqr @ dx0.reshape(1, 1)).reshape(-1)[0])

            # clip to input bounds
            if hasattr(self, "_du_min") and hasattr(self, "_du_max"):
                du0 = float(np.clip(du0, self._du_min, self._du_max))

            # update previous delta input for next observer step
            self._du_prev = np.array([[du0]], dtype=float)

            u0 = (us_col + np.array([[du0]], dtype=float)).reshape(-1)  # (1,)

            # provide trajectories
            x_traj = np.tile(y.reshape(-1, 1), (1, self.N + 1))          # (1,N+1) in absolute coords
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))             # (1,N) in absolute coords
            return u0, x_traj, u_traj

        # extract optimal trajectories 
        du_opt = np.array(self._du.value, dtype=float)                  # (1,N) delta
        dx_opt = np.array(self._dx.value, dtype=float)                  # (1,N+1) delta

        # update previous delta input 
        self._du_prev = du_opt[:, [0]]                                  # (1,1)

        # convert to absolute trajectories
        u_traj = self.us.reshape(-1, 1) + du_opt                         # (1,N)
        x_traj = self.xs.reshape(-1, 1) + dx_opt                         # (1,N+1)
        u0 = u_traj[:, 0].reshape(-1)              
        self._k_dbg = getattr(self, "_k_dbg", 0) + 1


        return u0, x_traj, u_traj
