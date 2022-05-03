import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.csv_trajectory import load_trajectory


class benchmarker():
    def __init__(self,
                 controller,
                 x0,
                 dt,
                 t_final,
                 goal,
                 integrator="runge_kutta",
                 save_dir="benchmark"):
        self.controller = controller
        self.x0 = np.asarray(x0)
        self.dt = dt
        self.t_final = t_final
        self.goal = np.asarray(goal)
        self.integrator = integrator
        self.save_dir = save_dir

        self.mass = None
        self.length = None
        self.com = None
        self.damping = None
        self.gravity = None
        self.cfric = None
        self.inertia = None
        self.motor_inertia = None
        self.torque_limit = None

        self.plant = None
        self.simulator = None
        self.ref_trajectory = None

        self.Q = None
        self.R = None
        self.Qf = None

        self.t_traj = None
        self.x_traj = None
        self.u_traj = None

        self.ref_cost_free = None
        self.ref_cost_tf = None

    def set_model_parameter(self,
                            mass,
                            length,
                            com,
                            damping,
                            gravity,
                            cfric,
                            inertia,
                            motor_inertia,
                            torque_limit):

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.gravity = gravity
        self.cfric = cfric
        self.inertia = inertia
        self.motor_inertia = motor_inertia
        self.torque_limit = torque_limit

        self.plant = SymbolicDoublePendulum(mass=mass,
                                            length=length,
                                            com=com,
                                            damping=damping,
                                            gravity=gravity,
                                            coulomb_fric=cfric,
                                            inertia=inertia,
                                            motor_inertia=motor_inertia,
                                            torque_limit=torque_limit)

        self.simulator = Simulator(plant=self.plant)

    def set_init_traj(self, trajectory_csv, read_with):
        self.t_traj, self.x_traj, self.u_traj = load_trajectory(trajectory_csv, read_with)

    def set_cost_par(self, Q, R, Qf):
        self.Q = Q
        self.R = R
        self.Qf = Qf

    def compute_cost(self, x_traj, u_traj, mode="free"):

        if mode == "free":
            X = x_traj[:-1] - self.goal
            U = u_traj
            xf = x_traj[-1] - self.goal
        elif mode == "trajectory_following":
            X = x_traj[:-1] - self.x_traj[:-1]
            U = u_traj - self.u_traj
            xf = x_traj[-1] - self.x_traj[-1]

        X_cost = np.einsum('jl, jk, lk', X.T, self.Q, X)
        U_cost = np.einsum('jl, jk, lk', U.T, self.R, U)
        Xf_cost = np.einsum('i, ij, j', xf, self.Qf, xf)

        cost = X_cost + U_cost + Xf_cost
        return cost

    def compute_ref_cost(self):
        self.ref_cost_free = self.compute_cost(self.x_traj, self.u_traj, mode="free")
        self.ref_cost_tf = self.compute_cost(self.x_traj, self.u_traj, mode="trajectory_following")

    def check_goal_success(self, x_traj, pos_eps=0.1, vel_eps=0.5):
        pos_succ = np.max(x_traj[-1][:2] - self.goal[:2]) < pos_eps
        vel_succ = np.max(x_traj[-1][2:] - self.goal[2:]) < vel_eps
        return (pos_succ and vel_succ)

    def compute_success_measure(self, x_traj, u_traj):
        X = np.asarray(x_traj)
        U = np.asarray(u_traj)
        cost_free = self.compute_cost(X, U, mode="free")
        cost_tf = self.compute_cost(X, U, mode="trajectory_following")
        succ = self.check_goal_success(X)
        return cost_free, cost_tf, succ

    def simulate_and_get_cost(self,
                              mass,
                              length,
                              com,
                              damping,
                              gravity,
                              cfric,
                              inertia,
                              motor_inertia,
                              torque_limit):

        plant = SymbolicDoublePendulum(mass=mass,
                                       length=length,
                                       com=com,
                                       damping=damping,
                                       gravity=gravity,
                                       coulomb_fric=cfric,
                                       inertia=inertia,
                                       motor_inertia=motor_inertia,
                                       torque_limit=torque_limit)

        simulator = Simulator(plant=plant)
        self.controller.init()

        T, X, U = simulator.simulate(t0=0., x0=self.x0, tf=self.t_final,
                                     dt=self.dt, controller=self.controller,
                                     integrator=self.integrator)

        cost_free, cost_tf, succ = self.compute_success_measure(X, U)
        return cost_free, cost_tf, succ

    def check_modelpar_robustness(
            self,
            mpar_vars=["Ir",
                       "m1r1", "I1", "b1", "cf1",
                       "m2r2", "m2", "I2", "b2", "cf2"],
            var_lists={"Ir": [],
                       "m1r1": [],
                       "I1": [],
                       "b1": [],
                       "cf1": [],
                       "m2r2": [],
                       "m2": [],
                       "I2": [],
                       "b2": [],
                       "cf2": []},
            ):

        print("computing model parameter robustness...")

        res_dict = {}
        for mp in mpar_vars:
            print("  ", mp)
            C_free = []
            C_tf = []
            SUCC = []
            for var in var_lists[mp]:
                if mp == "Ir":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=self.mass,
                            length=self.length,
                            com=self.com,
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=self.inertia,
                            motor_inertia=var,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "m1r1":
                    m1 = self.mass[0]
                    r1 = var/m1
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=[m1, self.mass[1]],
                            length=self.length,
                            com=[r1, self.com[1]],
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "I1":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=self.mass,
                            length=self.length,
                            com=self.com,
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=[var, self.inertia[1]],
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "b1":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=self.mass,
                            length=self.length,
                            com=self.com,
                            damping=[var, self.damping[1]],
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "cf1":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=self.mass,
                            length=self.length,
                            com=self.com,
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=[var, self.cfric[1]],
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "m2r2":
                    m2 = self.mass[1]
                    r2 = var/m2
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=[self.mass[0], m2],
                            length=self.length,
                            com=[self.com[0], r2],
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "m2":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=[self.mass[0], var],
                            length=self.length,
                            com=self.com,
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "I2":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=self.mass,
                            length=self.length,
                            com=self.com,
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=[self.inertia[0], var],
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "b2":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=self.mass,
                            length=self.length,
                            com=self.com,
                            damping=[self.damping[0], var],
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "cf2":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=self.mass,
                            length=self.length,
                            com=self.com,
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=[self.cfric[0], var],
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
            res_dict[mp] = {}
            res_dict[mp]["values"] = var_lists[mp]
            res_dict[mp]["free_costs"] = C_free
            res_dict[mp]["following_costs"] = C_tf
            res_dict[mp]["successes"] = SUCC
        return res_dict

    def check_perturbation_robustness(self,
                                      time_stamps=[],
                                      tau_perts=[]):
        pass

    def check_noise_robustness(self,
                               noise_mode="vel",
                               noise_amplitudes=[],
                               noise_cut=0.5,
                               noise_vfilter="lowpass",
                               noise_vfilter_args={"alpha": 0.3}):
        # maybe add noise frequency
        # (on the real system noise frequency seems so be higher than
        # control frequency -> no frequency neccessary here)
        print("computing noise robustness...")

        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        for na in noise_amplitudes:
            self.controller.init()
            self.simulator.set_imperfections(
                    noise_amplitude=na,
                    noise_mode=noise_mode,
                    noise_cut=noise_cut,
                    noise_vfilter=noise_vfilter,
                    noise_vfilter_args=noise_vfilter_args)
            T, X, U = self.simulator.simulate(
                    t0=0.0,
                    tf=self.t_final,
                    dt=self.dt,
                    x0=self.x0,
                    controller=self.controller,
                    integrator=self.integrator,
                    imperfections=True)
            self.simulator.reset_imperfections()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
        res_dict["noise_mode"] = noise_mode
        res_dict["noise_cut"] = noise_cut
        res_dict["noise_vfilter"] = noise_vfilter
        res_dict["noise_vfilter_args"] = noise_vfilter_args
        res_dict["noise_amplitudes"] = noise_amplitudes
        res_dict["free_costs"] = C_free
        res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        return res_dict

    def check_unoise_robustness(self,
                                unoise_amplitudes=[]):
        # maybe add noise frequency
        print("computing torque noise robustness...")

        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        for na in unoise_amplitudes:
            self.controller.init()
            self.simulator.set_imperfections(unoise_amplitude=na)
            T, X, U = self.simulator.simulate(
                    t0=0.0,
                    tf=self.t_final,
                    dt=self.dt,
                    x0=self.x0,
                    controller=self.controller,
                    integrator=self.integrator,
                    imperfections=True)
            self.simulator.reset_imperfections()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
        res_dict["unoise_amplitudes"] = unoise_amplitudes
        res_dict["free_costs"] = C_free
        res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        return res_dict

    def check_uresponsiveness_robustness(self,
                                         u_responses=[]):
        print("computing torque reponsiveness robustness...")

        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        for ur in u_responses:
            self.controller.init()
            self.simulator.set_imperfections(u_responsiveness=ur)
            T, X, U = self.simulator.simulate(
                    t0=0.0,
                    tf=self.t_final,
                    dt=self.dt,
                    x0=self.x0,
                    controller=self.controller,
                    integrator=self.integrator,
                    imperfections=True)
            self.simulator.reset_imperfections()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
        res_dict["u_reponsivenesses"] = u_responses
        res_dict["free_costs"] = C_free
        res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        return res_dict

    def check_delay_robustness(self,
                               delay_mode="posvel",
                               delays=[]):
        print("computing delay robustness...")
        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        for de in delays:
            self.controller.init()
            self.simulator.set_imperfections(
                    delay=de,
                    delay_mode=delay_mode)
            T, X, U = self.simulator.simulate(
                    t0=0.0,
                    tf=self.t_final,
                    dt=self.dt,
                    x0=self.x0,
                    controller=self.controller,
                    integrator=self.integrator,
                    imperfections=True)
            self.simulator.reset_imperfections()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
        res_dict["delay_mode"] = delay_mode
        res_dict["measurement_delay"] = delays
        res_dict["free_costs"] = C_free
        res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        return res_dict

    def benchmark(self,
                  compute_model_robustness=True,
                  compute_noise_robustness=True,
                  compute_unoise_robustness=True,
                  compute_uresponsiveness_robustness=True,
                  compute_delay_robustness=True,
                  mpar_vars=["Ir",
                             "m1r1", "I1", "b1", "cf1",
                             "m2r2", "m2", "I2", "b2", "cf2"],
                  modelpar_var_lists={"Ir": [],
                                      "m1r1": [],
                                      "I1": [],
                                      "b1": [],
                                      "cf1": [],
                                      "m2r2": [],
                                      "m2": [],
                                      "I2": [],
                                      "b2": [],
                                      "cf2": []},
                  noise_mode="vel",
                  noise_amplitudes=[0.1, 0.3, 0.5],
                  noise_cut=0.5,
                  noise_vfilter="lowpass",
                  noise_vfilter_args={"alpha": 0.3},
                  unoise_amplitudes=[0.1, 0.5, 1.0],
                  u_responses=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                  delay_mode="vel",
                  delays=[0.01, 0.02, 0.05, 0.1]):

        res = {}
        if compute_model_robustness:
            res_model = self.check_modelpar_robustness(
                    mpar_vars=mpar_vars,
                    var_lists=modelpar_var_lists)
            res["model_robustness"] = res_model
        if compute_noise_robustness:
            res_noise = self.check_noise_robustness(
                    noise_mode=noise_mode,
                    noise_amplitudes=noise_amplitudes,
                    noise_cut=noise_cut,
                    noise_vfilter=noise_vfilter,
                    noise_vfilter_args=noise_vfilter_args)
            res["noise_robustness"] = res_noise
        if compute_unoise_robustness:
            res_unoise = self.check_unoise_robustness(
                    unoise_amplitudes=unoise_amplitudes)
            res["unoise_robustness"] = res_unoise
        if compute_uresponsiveness_robustness:
            res_unoise = self.check_uresponsiveness_robustness(
                    u_responses=u_responses)
            res["unoise_robustness"] = res_unoise
        if compute_delay_robustness:
            res_delay = self.check_delay_robustness(
                    delay_mode=delay_mode,
                    delays=delays)
            res["delay_robustness"] = res_delay
        return res