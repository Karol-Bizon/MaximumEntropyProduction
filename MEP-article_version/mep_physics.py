# -*- coding: utf-8 -*-
"""
MEP Physics. All functions used by the minimize function are here.
"""

import numpy as np
import radiatif as rad  # with the python radiative code
# noinspection PyUnresolvedReferences
# from radiatif2 import Rad  # with the C++ radiative code
import profile_bis as prf
import physics as phy
import constants as cst


class MepPhysics:
    """
    Everything used inside the optimization loop (mainly scipy.optimize.minimize).
    """

    def __init__(self, parameters: dict[str, str | int | float | list | bool]):
        """
        :param parameters: dict[str, Any]
        """

        self.parameters = dict(parameters)

        self.alb = self.parameters["albedo"]
        self.n = self.parameters["number of levels"]
        self.physical_model = self.parameters['physical model']
        self.optimization_variable = self.parameters['optimization variable']

        self.Rad = rad.Radiation(self.n + 1, useRelativeH=True, useScaling='fixed')  # Syntax in python
        # self.radiaC = Rad(self.n, parameters['index of profile'], cst.p0, parameters['CO2'])  # Syntax in C++
        self.radiaC = self.Rad

        self.linearisation_radiativeflux = self.parameters['linearisation of radiative flux']

        self.pres = prf.pressureScale(cst.p0, self.n)  # pressure levels, middle of boxes = (ps, p1, p2 ...p_nLevels)
        self.presB = prf.pressureBounds(self.pres)

        self.G = self.def_G()

        dict_graph = self.createGraph()
        self.Grad = dict_graph["Grad"]
        self.Conv = dict_graph["Conv"]
        self.Trinf = dict_graph["Trinf"]
        self.N = dict_graph["N"]
        self.mu = self.N - self.n  # number of elementary cycles
        self.startMat = dict_graph["startMat"]
        self.targetMat = dict_graph["targetMat"]
        self.Cycl = dict_graph["Cycl"]

        dict_mass_variable = self.change_mass_variable()
        self.z_m = dict_mass_variable["z_m"]
        self.dmdz = dict_mass_variable["dmdz"]
        self.m_z = dict_mass_variable["m_z"]
        self.z_max = dict_mass_variable["z_max"]

        self.x_int = int("x" in self.optimization_variable)
        self.m_int = int("m" in self.optimization_variable)
        self.c_int = int("c" in self.optimization_variable)
        self.f_int = int("f" in self.optimization_variable)
        self.h_int = int("h" in self.optimization_variable)

        self.r = None
        self.r0 = None
        self.ri = None

        if 'feasibility objective function' in parameters:
            self.feasibility_objective_function = self.parameters['feasibility objective function']
            self.feasibility_variable = self.parameters["feasibility variable"]
        else:
            self.feasibility_objective_function = None
            self.feasibility_variable = None


    def def_G(self) -> np.ndarray:
        """
        See Appendix A of Labarre for the formula.
        Don't forget the layer 0 is an infinitely small boundary layer at atmospheric pressure, height=0.

        :return: G, 2D ndarray of size (n+1, n+1), corresponding to the isentropic geopotential matrix. height g z = GT
        """
        G = np.zeros((self.n + 1, self.n + 1))
        for i in range(1, self.n + 1):
            G[i, i] = cst.Cp * (np.power(self.presB[i - 1] / self.pres[i], cst.R / cst.Cp) - 1)
            for j in range(1, i):
                G[i, j] = cst.Cp * (np.power(self.presB[j - 1] / self.pres[j], cst.R / cst.Cp
                                             ) - np.power(self.presB[j] / self.pres[j], cst.R / cst.Cp))
        return G

    def change_mass_variable(self) -> dict:
        """
        :return: dict{z_m: Callable, dmdz: Callable, m_z: Callable, z_max: float | None}

        z_m is z as a function of m
        m_z is m as a function of z
        dmdz is the derivative
        z_max = z_m("maximal mass")
        """
        if 'm' in self.optimization_variable:

            if self.parameters['function of the variable change'] == '1-exp':
                m_ref = self.parameters['mass reference']
                p_ref = self.parameters['percent reference']
                coef_mul = self.parameters['coefficient multiplication']
                coef_add = self.parameters['coefficient addition']
                coef_ref = m_ref / (-np.log(1 - p_ref))

                def z_m(m: np.ndarray | float) -> np.ndarray | float:
                    return coef_add + coef_mul * (1 - np.exp(-m / coef_ref))

                def m_z(z: np.ndarray | float) -> np.ndarray | float:
                    return - coef_ref * np.log((1 - (z - coef_add) / coef_mul))

                def dmdz(z: np.ndarray | float) -> np.ndarray | float:
                    return coef_ref / coef_mul * np.divide(1, 1 - ((z - coef_add) / coef_mul))

            elif self.parameters['function of the variable change'] == 'tanh':
                m_ref = self.parameters['mass reference']
                p_ref = self.parameters['percent reference']
                coef_mul = self.parameters['coefficient multiplication']
                coef_add = self.parameters['coefficient addition']
                coef_ref = m_ref / np.arctanh(p_ref)

                def z_m(m: np.ndarray | float) -> np.ndarray | float:
                    return coef_add + coef_mul * np.tanh(m / coef_ref)

                def m_z(z: np.ndarray | float) -> np.ndarray | float:
                    return coef_ref * np.arctanh((z - coef_add) / coef_mul)

                def dmdz(z: np.ndarray | float) -> np.ndarray | float:
                    return coef_ref / coef_mul * np.divide(1, 1 - np.power((z - coef_add) / coef_mul, 2))

            else:
                if self.parameters['function of the variable change'] == 'no':
                    coef_mul = 1
                else:
                    coef_mul = self.parameters['coefficient multiplication']

                def z_m(m: np.ndarray | float) -> np.ndarray | float:
                    return coef_mul * m

                def m_z(z: np.ndarray | float) -> np.ndarray | float:
                    return 1 / coef_mul * z

                def dmdz(_z: np.ndarray | float) -> np.ndarray | float:
                    return 1 / coef_mul * np.ones(self.N)

            m_max = self.parameters['maximal mass']
            z_max = z_m(m_max)

        else:
            coef_mul = 1

            def z_m(m: np.ndarray | float) -> np.ndarray | float:
                return coef_mul * m

            def m_z(z: np.ndarray | float) -> np.ndarray | float:
                return 1 / coef_mul * z

            def dmdz(_z: np.ndarray | float) -> np.ndarray | float:
                return 1 / coef_mul * np.ones(self.N)

            z_max = None

        return {"z_m": z_m, "dmdz": dmdz, "m_z": m_z, "z_max": z_max}

    # function to extract variables
    def extract_variable(self, v: np.ndarray) -> tuple:
        """
        In the optimization problem, the variable v is used. But depending on the parameters, it doesn't contain every
        possible variable. It contains an assortment of (x, z, c, f and h).

        :param v: 1D ndarray, variable of the optimization problem, concatenation of (z(m), c, f, and/or h)
        :return (x, z, x, f, h): tuple of 1D or empty ndarray
        """
        x = v[:self.x_int * (self.n + 1)]
        z = v[self.x_int * (self.n + 1):self.x_int * (self.n + 1) + self.m_int * self.N]
        c = v[self.x_int * (self.n + 1) + self.m_int * self.N:
              self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu]
        f = v[self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu:
              self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N]
        h = v[self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N:
              self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N
              + self.h_int * (self.n + 1)]
        return x, z, c, f, h

    @staticmethod
    def hs(x: np.ndarray) -> np.ndarray:
        """
        Saturation vapour pressure of water vapor in air (in hPa) (similar to Magnus or Bolton formula).

        :param x: 1D ndarray of size (n+1), Tref/T
        :return: 1D ndarray of size (n+1). P_s = a expt[bT / (c + T)] (hPa)
        """
        return phy.ew_vec(cst.Tref / x)

    @staticmethod
    def qs(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Specific humidity of water vapor in air at saturation (m_water / m_air).

        :param x: 1D ndarray of size (n+1), cst.Tref/T
        :param p: 1D ndarray of size (n+1), pressure in hPa

        :return: 1D ndarray of size (n+1).
        """
        return phy.rsat_vec(p, cst.Tref / x)

    def energy_xq(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:  # Energy profile, depending on the constraint
        """
        If "un",    e plays no role in the optimization problem, but be computed anyway
        If "cpT":   e = Cp T
        If "dry":   e = Cp T + g z
        If "moist": e = Cp T + g z + L q_s
        If "wc":    e = Cp T + g z + L q

        :param x: 1D ndarray, x = Tref/T
        :param h: 1D ndarray, relative humidity h = q / q_s
        :return: 1D ndarray of dim (n+1), the energy
        """
        invx = 1 / x
        if "h" not in self.optimization_variable:
            h = 1
        if 'cpT' in self.physical_model:  # Thermal energy if no constraint
            E = cst.Cp * cst.Tref * invx
        elif 'dry' in self.physical_model:  # Dry energy = thermal energy + geopotential
            E = cst.Cp * cst.Tref * invx + np.dot(self.G, cst.Tref * invx)
        elif ("moist" in self.physical_model) or ("wc" in self.physical_model) or ("un" in self.physical_model):
            # Moist energy = dry energy + latent heat (all the possible component)
            E = cst.Cp * cst.Tref * invx + np.dot(self.G, cst.Tref * invx) + cst.La * h * self.qs(x, self.pres)
        else:
            raise Exception("Probably a typo in parameter \"physical modeRl\"")
        e = E / cst.Eref
        return e

    def energy_dry(self, x: np.ndarray) -> np.ndarray:
        """
         Used only for ini_moist and wc.
        :param x: 1D ndarray, x = Tref/T
        :return: e_d = Cp T + g z
        """
        invx = 1 / x
        E = cst.Cp * cst.Tref * invx + np.dot(self.G, cst.Tref * invx)
        e_dry = E / cst.Eref

        return e_dry

    def energy_wet(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Used only for ini_moist and wc.

        :param x: 1D ndarray, x = Tref/T
        :param h: 1D ndarray, relative humidity h = q / q_s

        :return: e_w = L q
        """
        if "h" not in self.optimization_variable:
            h = 1
        E = cst.La * h * self.qs(x, self.pres)

        e_wet = E / cst.Eref

        return e_wet

    def Latent_heat_derivative_x(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: 1D ndarray, x = Tref/T
        :return: d/dx ( L q(x) )
        """
        dTdx = -cst.Tref / np.power(x, 2)
        dqdT = phy.drsat_dT_vec(self.pres, cst.Tref / x)
        dLhdx = cst.La * dqdT * dTdx
        return dLhdx

    def energy_jac_x(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        :param x: 1D ndarray, x = Tref/T
        :param h: 1D ndarray, relative humidity h = q / q_s

        :return: 2D ndarray of dim (n+1, n+1), (de_i/dx_j)
        """
        dTdx = -cst.Tref / np.power(x, 2)

        if "h" not in self.optimization_variable:
            h = 1

        if ('cpT' in self.physical_model) or ('un' in self.physical_model):  # Thermal energy if no
            # constraint
            dEdT = cst.Cp * np.eye(self.n + 1)
            dEdx = dTdx * dEdT
        elif 'dry' in self.physical_model:
            dEdT = cst.Cp * np.eye(self.n + 1) + self.G
            dEdx = dTdx * dEdT
        else:
            dEdx = dTdx * (cst.Cp * np.eye(self.n + 1) + self.G) + np.diag(h * self.Latent_heat_derivative_x(x))

        dedx = dEdx / cst.Eref

        return dedx

    def x_f(self, f: np.ndarray) -> np.ndarray:
        """
        This function works only when radiative fluxes are linearised.

        rf(x) = rx + r0  linearisation
        rf(x) = Div(f)   energy conservation

        :param f: 1D ndarray of dim N.
        :return: x, 1D ndarray of dim (n+1). x = r^-1 (Div(f) - r0)
        """
        fconmr0 = (- self.r0 - np.dot(self.Conv, f))
        x = np.dot(self.ri, fconmr0)
        return x

    def param_radiative_flux(self, x: np.ndarray) -> None:
        """
        r, r0 and ri (=r^-1) are computed here.
        rf(x) = b = rx + r0, is the radiative flux

        It is used in 2 cases: 1) to linearize rf(x) before an optimization loop (resolution iteration).
                               2) to compute the jacobian of rf(x) when there is no linearisation of rf.

        :param x: 1D ndarray of dim (n+1).
        """

        # Syntax in python
        self.r, self.r0, b = self.Rad.ddx_bilanR(x)
        # Syntax in C++
        # b, self.r = self.radiaC.ddx_bilanR(x)
        # self.r0 = b - np.dot(self.r, x)

        self.ri = np.linalg.inv(self.r)

    def radiative_flux(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: 1D ndarray of dim (n+1).
        :return: rf, 1D ndarray of dim (n+1), the radiative flux
        """
        if self.linearisation_radiativeflux == "once per resolution iteration":
            rf = np.dot(self.r, x) + self.r0

        elif self.linearisation_radiativeflux == "no linearisation":
            # Syntax in python
            # rf = self.Rad.bilanR(x)
            # Syntax in C++
            rf = self.radiaC.bilanR(x)

        else:
            print('The method for the linearisation of the radiative fluxes has to be precised')
            rf = np.dot(self.r, x) + self.r0

        return rf

    def objective_function_feasibility(self, v: np.ndarray) -> float:
        """
        The function to minimize to test the feasibility of the constraints.
        Options are: null function, sum v, - sum v, sum (v - v0) ** 2

        :param v: 1D ndarray, concatenation of optimization variables
        :return: float, the objective function
        """
        x, z, c, f, h = self.extract_variable(v)
        feas_var = self.parameters["feasibility variable"]
        v_cut = np.concatenate((x if "x" in feas_var else np.empty(0), z if "m" in feas_var else np.empty(0),
                                c if "c" in feas_var else np.empty(0), f if "f" in feas_var else np.empty(0),
                                h if "h" in feas_var else np.empty(0)))

        if self.feasibility_objective_function == 'null':
            return 0
        elif self.feasibility_objective_function == 'minimal sum':
            return sum(v_cut)
        elif self.feasibility_objective_function == 'maximal sum':
            return - sum(v_cut)
        elif self.feasibility_objective_function == 'sum equal':
            value_to_equal = self.parameters['value to be equal to']
            return sum((v_cut - value_to_equal) ** 2)

    def objective_function_feasibility_jac(self, v: np.ndarray) -> np.ndarray:
        """
        The jacobian of the function to minimize to test the feasibility of the constraints.

        :param v: 1D ndarray, concatenation of optimization variables
        :return: 1D ndarray, the jacobian of the objective function
        """
        x, z, c, f, h = self.extract_variable(v)
        x_i = int("x" in self.feasibility_variable)
        m_i = int("m" in self.feasibility_variable)
        c_i = int("c" in self.feasibility_variable)
        f_i = int("f" in self.feasibility_variable)
        h_i = int("h" in self.feasibility_variable)
        feas_var = self.parameters["feasibility variable"]
        opt_var = self.optimization_variable
        v_cut = np.concatenate((x if "x" in feas_var else np.empty(0), z if "m" in feas_var else np.empty(0),
                                c if "c" in feas_var else np.empty(0), f if "f" in feas_var else np.empty(0),
                                h if "h" in feas_var else np.empty(0)))
        len_v_cut = len(v_cut)

        if self.feasibility_objective_function == 'null':
            jac_v_cut = np.zeros(len_v_cut)
        if self.feasibility_objective_function == 'minimal sum':
            jac_v_cut = np.ones(len_v_cut)
        if self.feasibility_objective_function == 'maximal sum':
            jac_v_cut = - np.ones(len_v_cut)
        if self.feasibility_objective_function == 'sum equal':
            value_to_equal = self.parameters['value to be equal to']
            jac_v_cut = 2 * (v_cut - value_to_equal)

        jac_x, jac_z, jac_c, jac_f, jac_h = (np.empty(0),) * 5
        if "x" in opt_var:
            jac_x = np.zeros(self.n + 1)
            if "x" in feas_var:
                # noinspection PyUnboundLocalVariable
                jac_x = jac_v_cut[:x_i * (self.n + 1)]
        if "m" in opt_var:
            jac_z = np.zeros(self.N)
            if "m" in feas_var:
                jac_z = jac_v_cut[x_i * (self.n + 1): x_i * (self.n + 1) + m_i * self.N]
        if "c" in opt_var:
            jac_c = np.zeros(self.mu)
            if "c" in feas_var:
                jac_c = jac_v_cut[x_i * (self.n + 1) + m_i * self.N: x_i * (self.n + 1) + m_i * self.N + c_i * self.mu]
        if "f" in opt_var:
            jac_f = np.zeros(self.N)
            if "f" in feas_var:
                jac_f = jac_v_cut[x_i * (self.n + 1) + m_i * self.N + c_i * self.mu:
                                  x_i * (self.n + 1) + m_i * self.N + c_i * self.mu + f_i * self.N]
        if "h" in opt_var:
            jac_h = np.zeros(self.n + 1)
            if "h" in feas_var:
                jac_h = jac_v_cut[x_i * (self.n + 1) + m_i * self.N + c_i * self.mu + f_i * self.N:
                                  x_i * (self.n + 1) + m_i * self.N + c_i * self.mu + f_i * self.N + h_i * (self.n + 1)]

        jac = np.concatenate((jac_x, jac_z, jac_c, jac_f, jac_h))

        return jac

    def maximum_entropyprod(self) -> object:
        """
        For the unconstrained case.
        The optimization problem is written with a Lagrangian formulation (method of Lagrange multipliers)
        and the radiative transfer rf is linearized so that the problem becomes linear.

        We solve :
          | (r+rT)   -sRT |.|x| = |-r0
          |  -sR       0  | |ß|   |sR0

        where L = σ(x) - ß c(x) = -xT rf(x) - ß Σ rf(x)
              rf(x) = rx + r0
              sR_(j) = Σ_(i) r_(ij)
              sR0 = Σ_(i) R0_(i)

        :return: res (resolution)
        """
        sr = -np.dot(np.ones(self.n + 1), self.r)
        m = np.append(self.r, np.reshape(sr, (1, self.n + 1)),
                      axis=0)  # np.append(np.append(r, sr, axis=1),np.zeros(n+2),axis=0)
        m = np.append(m, np.zeros((self.n + 2, 1)), axis=1)
        m = m + np.transpose(m)
        b = np.append(-self.r0, np.sum(self.r0))
        # print('b=',b)
        sol = np.linalg.solve(m, b)
        v = sol[:-1]
        rf = self.radiative_flux(v)
        entropy_prod = np.dot((-1) * rf, v)

        # I create a class just to be coherent with the code with scipy.optimize
        class Optimize:
            def __init__(self, fun, success, x):
                self.fun = fun
                self.success = success
                self.x = x

        res = Optimize(-entropy_prod, True, v)
        return res

    # Function of f
    def minus_entropyprod_f(self, v: np.ndarray) -> float:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (z(m), c, f, and/or h)
                  It contains at least f.
        :return: float, the opposite of entropy production (depends on f only)
        """
        _, _, _, f, _ = self.extract_variable(v)

        x = self.x_f(f)  # x = (r^-1)*(Conv(F)-r0) != x_ini used to calculate r_ini,r0_ini

        entropy_prod = np.linalg.multi_dot([x, self.Conv, f])

        return - entropy_prod

    def minus_entropyprod_jac_f(self, v: np.ndarray) -> np.ndarray:
        """
            :param v: 1D ndarray, variable of the optimization problem, concatenation of (z(m), c, f, and/or h)
            :return: 1D ndarray, jacobian of opposite entropy production (depends on f only)
            """
        _, _, _, f, _ = self.extract_variable(v)

        jac = np.zeros(len(v))
        x = self.x_f(f)
        jac_f = np.linalg.multi_dot([f, np.transpose(self.Conv), self.ri, -self.Conv]) + np.dot(x, self.Conv)

        jac[self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu:
            self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N] = jac_f

        return - jac

    # Function of xm
    def minus_entropyprod_xz(self, v: np.ndarray) -> float:
        """
        :param v: 1D ndarray, concatenation of optimization variables, containing at least x and m.
        :return: 1D ndarray, opposite of entropy production (depends on x and m).
        """
        x, z, _, _, h = self.extract_variable(v)

        e = self.energy_xq(x, h)

        entropy_prod = np.linalg.multi_dot([x, self.Conv, self.m_z(z)[:, None] * -self.Grad, e])

        return - entropy_prod

    def minus_entropyprod_jac_xz(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, concatenation of optimization variables, containing at least x and m.
        :return: 1D ndarray, jacobian of opposite entropy production (depends on x and m)
        """
        x, z, _, _, h = self.extract_variable(v)

        jac = np.zeros(len(v))

        e = self.energy_xq(x, h)

        # jacobian of x
        dedx = self.energy_jac_x(x, h)

        jac_x = np.linalg.multi_dot([self.Conv, self.m_z(z)[:, None] * -self.Grad, e]) + \
                np.linalg.multi_dot([x, self.Conv, self.m_z(z)[:, None] * -self.Grad, dedx])

        # jacobian of z
        delta_e = np.dot(-self.Grad, e)
        dfdz = np.diag(self.dmdz(z) * delta_e)
        jac_z = np.linalg.multi_dot([x, self.Conv, dfdz])

        jac[:self.n + 1 + self.N] = np.concatenate((jac_x, jac_z))

        if "h" in self.optimization_variable:
            # jacobian of h
            dedh = np.diag(cst.La * self.qs(x, self.pres)) / cst.Eref
            jac_h = np.linalg.multi_dot([x, self.Conv, self.m_z(z)[:, None] * -self.Grad, dedh])
            jac[self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N:
                self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N +
                self.h_int * (self.n + 1)] = jac_h  # x_int = m_int = 1

        return - jac

    # Function of x
    def minus_entropyprod_x(self, v: np.ndarray) -> float:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It contains at least x.
        :return: float, the opposite of entropy production (depends on x only)
        """
        x = v[: self.n + 1]

        rf = self.radiative_flux(x)  # use r and r0 with "once per resolution iteration"
        entropy_prod = np.dot((-1) * rf, x)

        return - entropy_prod

    def minus_entropyprod_jac_x(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
        :return: 1D ndarray, jacobian of opposite entropy production (depends on x only)
        """
        x, _, _, _, _ = self.extract_variable(v)

        if self.linearisation_radiativeflux == "no linearisation":
            self.param_radiative_flux(x)  # modifies r and r0

        jac = np.zeros(len(v))
        jac[:self.n + 1] = np.dot(self.r + np.transpose(self.r), x) + self.r0

        return jac

    def minus_entropyprod_hess_x(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
        :return: 2D ndarray, hessian of opposite entropy production
        """
        x, _, _, _, _ = self.extract_variable(v)

        if self.linearisation_radiativeflux == "no linearisation":
            self.param_radiative_flux(x)  # modifies r and r0

        hess = np.zeros((len(v), len(v)))
        hess[:self.n + 1, :self.n + 1] = (self.r + np.transpose(self.r))

        return hess

    # Function of xf
    def minus_entropyprod_xf(self, v: np.ndarray) -> float:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It contains at least x and f.
        :return: float, the opposite of entropy production
        """
        x, _, _, f, _ = self.extract_variable(v)

        entropy_prod = np.linalg.multi_dot([x, self.Conv, f])

        return - entropy_prod

    def minus_entropyprod_jac_xf(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It contains at least x and f.
        :return: 1D ndarray, jacobian of opposite entropy production (depends on x and f)
        """
        x, _, _, f, _ = self.extract_variable(v)

        jac = np.zeros(len(v))

        jac_x = np.dot(self.Conv, f)
        jac_f = - np.dot(-self.Grad, x)

        jac[:(self.n + 1)] = jac_x
        jac[(self.n + 1) + self.m_int * self.N + self.c_int * self.mu:
            (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.N] = jac_f

        return - jac

    def con_pos_m(self, v):  # m > 0 (but con_pos_alpha is used instead)
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, f, and/or h)
                  It contains at least x or f. It doesn't contain z(m).
        :return: 1D ndarray of dim=dim(m), constraint m = f / Grad(e(x)) >= 0
        :To avoid division by Grad(e(x)), we prefer using con_pos_alpha instead
        """
        x, _, _, f, h = self.extract_variable(v)

        if "x" in self.optimization_variable and "f" not in self.optimization_variable:
            rf = self.radiative_flux(x)
            f = np.dot(self.Trinf, rf)

        elif "f" in self.optimization_variable and "x" not in self.optimization_variable:
            x = self.x_f(f)

        e = self.energy_xq(x, h)
        delta_e = np.dot(-self.Grad, e)
        m = np.divide(f, delta_e)

        return m

    def con_pos_alpha(self, v: np.ndarray) -> np.ndarray:  # m > 0 expressed as grad(F)*e > 0
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, f, and/or h)
                  It contains at least x or f. It doesn't contain z(m).
        :return: 1D ndarray of dim=dim(f), constraint alpha = F*Grad(e(x)) >= 0
        """
        x, _, _, f, h = self.extract_variable(v)

        if "x" not in self.optimization_variable:
            x = self.x_f(f)

        if "f" not in self.optimization_variable:
            rf = self.radiative_flux(x)
            f = np.dot(self.Trinf, rf)

        e = self.energy_xq(x, h)
        delta_e = np.dot(-self.Grad, e)  # delta_e = e[i] - e[i+1]
        alpha = f * delta_e

        return alpha  # - 1e-9; #minus epsilon to avoid m < 0 when delta_e=0

    def con_pos_alpha_jac(self,
                          v: np.ndarray) -> np.ndarray:  # jacobian of the constraint m > 0 expressed as grad(F)*e > 0
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, f, and/or h)
                  It contains at least x or f. It doesn't contain z(m).
        :return: 2D ndarray of dim=(dim(f),dim(v)) jacobian of constraint F*Grad(e(x)) >= 0
        """
        x, _, _, f, h = self.extract_variable(v)

        if "x" in self.optimization_variable and "f" in self.optimization_variable:
            dfdx = np.zeros((self.N, self.n + 1))
            dxdf = np.zeros((self.n + 1, self.N))
        elif "f" in self.optimization_variable and "x" not in self.optimization_variable:
            x = self.x_f(f)
            dxdf = np.dot(self.ri, -self.Conv)
            # dedx = np.zeros(N)
        elif "x" in self.optimization_variable and "f" not in self.optimization_variable:
            if self.linearisation_radiativeflux == "no linearisation":
                self.param_radiative_flux(x)  # modifies r and r0
            rf = self.radiative_flux(x)
            f = np.dot(self.Trinf, rf)
            dfdx = np.dot(self.Trinf, self.r)

        jac = np.zeros((self.N, len(v)))

        e = self.energy_xq(x, h)
        delta_e = np.dot(-self.Grad, e)
        dedx = self.energy_jac_x(x, h)

        if "x" in self.optimization_variable:
            # noinspection PyUnboundLocalVariable
            jac_x = dfdx * delta_e[:, None] + np.dot(f[:, None] * -self.Grad, dedx)

            jac[:, :self.x_int * (self.n + 1)] = jac_x

        if "f" in self.optimization_variable:
            # noinspection PyUnboundLocalVariable
            jac_f = np.diag(delta_e) + np.linalg.multi_dot([f[:, None] * -self.Grad, dedx, dxdf])

            jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu:
                   self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N] = jac_f

        if "h" in self.optimization_variable:
            dedh = np.diag(cst.La * self.qs(x, self.pres)) / cst.Eref
            # print("dedh", dedh)
            jac_h = np.dot(f[:, None] * -self.Grad, dedh)

            jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N:
                   self.x_int * (
                           self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N + self.h_int * (
                           self.n + 1)] = jac_h
        return jac

    def con_pos_alpha_error(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, f, and/or h)
                  It contains at least x or f. It doesn't contain z(m).
        :return: 1D ndarray, the "error" on the constraint alpha, computed as max(-alpha, 0)_i/mean(|alpha|)
        """
        alpha = self.con_pos_alpha(v)

        alpha_err = np.maximum(-alpha, np.zeros(alpha.shape)) / np.mean(np.abs(alpha))

        return alpha_err

    # Constraint on the global energy fluxes balance

    def con_global_energy_balance(self, v: np.ndarray) -> float:
        """
        :param v: 1D ndarray, variable of the optimization problem, should be (x and/or h)
                  It must contain x.
        :return: float, a constraint we want equal to zero (sum(rf(x)) = 0)
        """
        x, _, _, _, _ = self.extract_variable(v)

        rf = self.radiative_flux(x)
        s = float(np.sum(rf))

        return s

    def con_global_energy_balance_jac(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, should be (x and/or h)
                  It must contain x.
        :return: 1D ndarray, jacobian of the constraint sum(rf(x)) = 0 (i.e [sum_i r_ij] j=0,...,n+1)
        """
        jac = np.zeros(len(v))
        if self.linearisation_radiativeflux == "no linearisation":
            x = v[:self.n + 1]
            self.param_radiative_flux(x)  # modifies r and r0
        jac[:self.x_int * (self.n + 1)] = np.sum(self.r, axis=0)

        return jac

    def con_global_energy_balance_error(self, v: np.ndarray) -> float:  # sum(rf(i))/sum(|rf(i)|)
        """
        :param v: 1D ndarray, variable of the optimization problem, should be (x and/or h)
                  It must contain x.
        :return: float, the "relative error" on the constraint, computed as: sum(rf(x)) / sum(|rf(x)|)
        """
        x, _, _, _, _ = self.extract_variable(v)

        rf = self.radiative_flux(x)
        s = np.sum(rf)
        s_abs = np.sum(np.abs(rf))

        return s / s_abs

    # Constraint on the energy flux balance for a box
    def con_local_energy_balance(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain x and (m, c or f).
        :return: 1D ndarray, the constraint rf(x) = Div(f)
        """
        x, z, c, f, h = self.extract_variable(v)

        if "f" not in self.optimization_variable:
            f = np.zeros(self.N)
            e = self.energy_xq(x, h)
            if "m" in self.optimization_variable:
                f += np.dot(self.m_z(z)[:, None] * -self.Grad, e)
            if "c" in self.optimization_variable:
                f += self.transport(c, e)

        rf = self.radiative_flux(x)
        cons = rf + np.dot(self.Conv, f)

        # if self.optimization_variable == 'xmh':
        #     x, z, h = v[:n + 1], v[n + 1:N + n + 1], v[N + n + 1:]
        #     q = h * self.qs(x, self.pres)
        #     e = self.energy_xq(x, q)
        #     f = convective_flux_ez(e, z)

        return cons

    def con_local_energy_balance_jac(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain x and (m, c or f).
        :return: 2D ndarray of dim=(dim(x), dim(v)) jacobian of the constraint rf(x) = Div(f)
        """
        x, z, c, _, h = self.extract_variable(v)

        jac = np.zeros((self.n + 1, len(v)))

        if self.linearisation_radiativeflux == "no linearisation":
            self.param_radiative_flux(x)  # modifies r and r0

        jac_x = self.r

        if "f" in self.optimization_variable:
            jac_f = self.Conv

            jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu:
                   self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N] = jac_f
        else:
            e = self.energy_xq(x, h)
            dedx = self.energy_jac_x(x, h)
            if "m" in self.optimization_variable:
                # jac x
                jac_x += np.linalg.multi_dot([self.Conv, self.m_z(z)[:, None] * -self.Grad, dedx])

                # jac z
                delta_e = np.dot(-self.Grad, e)
                dfdz = np.diag(self.dmdz(z) * delta_e)
                jac_z = np.dot(self.Conv, dfdz)

                jac[:, self.x_int * (self.n + 1): self.x_int * (self.n + 1) + self.m_int * self.N] = jac_z

            if "c" in self.optimization_variable:
                # jac_x
                jac_x += np.dot(self.Conv, self.transport_jac_x(c, dedx))

                # jac_c
                jac_c = np.dot(self.Conv, self.transport_jac_a(c, e))

                jac[:, self.x_int * (self.n + 1) + self.m_int * self.N:
                       self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu] = jac_c

            if "h" in self.optimization_variable:
                # jac of h
                jac_h = np.zeros((self.n + 1, self.n + 1))
                dedh = np.diag(cst.La * self.qs(x, self.pres)) / cst.Eref
                if "m" in self.optimization_variable:
                    jac_h += np.linalg.multi_dot([self.Conv, self.m_z(z)[:, None] * -self.Grad, dedh])
                if "c" in self.optimization_variable:
                    jac_h += np.dot(self.Conv, self.transport_jac_x(c, dedh))

                jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N:
                       self.x_int * (
                               self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N + self.h_int * (
                               self.n + 1)] = jac_h

        jac[:, :self.x_int * (self.n + 1)] = jac_x

        return jac

    def con_local_energy_balance_error(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain x and (m or f).
        :return: 1D ndarray of dim=dim(x) the "relative error" on the constraint,
                 computed as: (rf(x) - Div(f)) / mean(|rf(x)|)
        """
        x, _, _, _, _ = self.extract_variable(v)

        rf = self.radiative_flux(x)
        cons = self.con_local_energy_balance(v)
        error = np.divide(cons, np.mean(np.abs(rf)))

        return error

    # Constraint to express the relation between f and m and e

    def con_def_convective_flux(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain f and (m and/or c).
        :return: 1D ndarray of dim=dim(f), the constraint f = m*Grad(e) + transport(c, e)
        """
        x, z, c, f, h = self.extract_variable(v)

        cons = -f
        if "x" not in self.optimization_variable:
            x = self.x_f(f)
        e = self.energy_xq(x, h)
        if "m" in self.optimization_variable:
            cons += self.m_z(z) * np.dot(-self.Grad, e)

        if "c" in self.optimization_variable:
            cons += self.transport(c, e)

        return cons

    def con_def_convective_flux_jac(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain f and (m and/or c).
        :return: 2D ndarray of dim=(dim(f), dim(v)), jacobian of the constraint f = m*Grad(e) + transport(c, e)
        """
        x, z, c, f, h = self.extract_variable(v)

        jac = np.zeros((self.N, len(v)))

        jac_f = - np.identity(self.N)

        if "x" in self.optimization_variable:
            jac_x = np.zeros((self.N, self.n + 1))
            dedx = self.energy_jac_x(x, h)
            if "m" in self.optimization_variable:
                jac_x += self.m_z(z)[:, None] * np.dot(-self.Grad, dedx)
            if "c" in self.optimization_variable:
                jac_x += self.transport_jac_x(c, dedx)

            jac[:, :self.x_int * (self.n + 1)] = jac_x
        else:
            x = self.x_f(f)
            dedx = self.energy_jac_x(x, h)
            dxdf = np.dot(self.ri, -self.Conv)
            ddelta_edf = np.linalg.multi_dot([-self.Grad, dedx, dxdf])
            if "m" in self.optimization_variable:
                jac_f += self.m_z(z)[:, None] * ddelta_edf
            if "c" in self.optimization_variable:
                jac_f += np.dot(self.transport_jac_x(c, dedx), dxdf)

        jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu:
               self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N] = jac_f

        e = self.energy_xq(x, h)
        if "m" in self.optimization_variable:
            jac_z = np.diag(self.dmdz(z) * np.dot(-self.Grad, e))

            jac[:, self.x_int * (self.n + 1):self.x_int * (self.n + 1) + self.m_int * self.N] = jac_z
        if "c" in self.optimization_variable:
            jac_c = self.transport_jac_a(c, e)

            jac[:, self.x_int * (self.n + 1) + self.m_int * self.N:
                   self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu] = jac_c

        if "h" in self.optimization_variable:
            jac_h = np.zeros((self.N, self.n + 1))
            dedh = np.diag(cst.La * self.qs(x, self.pres)) / cst.Eref
            if "m" in self.optimization_variable:
                jac_h += self.m_z(z)[:, None] * np.dot(-self.Grad, dedh)
            if "c" in self.optimization_variable:
                jac_h += self.transport_jac_x(c, dedh)

            jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N:
                   self.x_int * (
                           self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N + self.h_int * (
                           self.n + 1)] = jac_h

        return jac

    def con_def_convective_flux_error(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain f and (m and/or c).
        :return: 1D ndarray, the "relative error" computed as (f - m*Grad(e) - transport(c,e)) / mean(|f|)
        """
        _, _, _, f, _ = self.extract_variable(v)

        cons = self.con_def_convective_flux(v)
        error = np.divide(cons, np.mean(np.abs(f)))
        return error

    def con_pos_p(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain x or f.
        :return: 1D ndarray of dim=dim(f), the constraint P = +Div(m*Grad(q) + transport(c, q))
        """
        x, z, c, f, h = self.extract_variable(v)

        m = self.m_z(z)
        if "f" in self.optimization_variable and "x" not in self.optimization_variable:
            x = self.x_f(f)
        if "h" not in self.optimization_variable:
            h = 1
        if "m" not in self.optimization_variable and "c" not in self.optimization_variable:
            if "x" in self.optimization_variable and "f" not in self.optimization_variable:
                if self.linearisation_radiativeflux == "no linearisation":  # we might not need these 2 lines.
                    self.param_radiative_flux(x)  # modifies r and r0
                rf = self.radiative_flux(x)
                f = np.dot(self.Trinf, rf)

            e = self.energy_xq(x, h)
            delta_e = np.dot(-self.Grad, e)
            m = np.divide(f, delta_e)

        q = h * self.qs(x, self.pres)

        p_temp = np.zeros(self.n + 1)

        if "c" in self.optimization_variable:
            p_temp += np.dot(self.Conv, self.transport(c, q))
        if "m" in self.optimization_variable or "c" not in self.optimization_variable:
            p_temp += np.linalg.multi_dot([self.Conv, m[:, None] * -self.Grad, q])

        p = p_temp[1:] * cst.Lref

        return p

    def con_pos_p_jac(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain x or f.
        :return: 2D ndarray of dim=(dim(x) - 1, dim(v)), the jacobian of the constraint P = +Div(m*Grad(q) + transport(c, q))
        :!: If m not in optimization variable but is expressed as a function of x and f, I didn't code the derivative dmdh.
        So if you put h, please also put m or c in optimization variable.
        """
        x, z, c, f, h = self.extract_variable(v)

        jac = np.zeros((self.n + 1, len(v)))

        m = self.m_z(z)
        if "x" in self.optimization_variable:
            dxdf = np.zeros((self.n + 1, self.N))
        elif "f" in self.optimization_variable and "x" not in self.optimization_variable:
            x = self.x_f(f)
            dxdf = np.dot(self.ri, -self.Conv)
        if "h" not in self.optimization_variable:
            h = 1
        if "m" in self.optimization_variable or "c" in self.optimization_variable:
            dmdx = np.zeros((self.N, self.n + 1))
            dmdf = np.zeros((self.N, self.N))
            if "m" not in self.optimization_variable:
                m = np.zeros(self.N)
        else:
            if "x" in self.optimization_variable and "f" not in self.optimization_variable:
                if self.linearisation_radiativeflux == "no linearisation":
                    self.param_radiative_flux(x)  # modifies r and r0
                rf = self.radiative_flux(x)
                f = np.dot(self.Trinf, rf)
                dfdx = np.dot(self.Trinf, self.r)
                e = self.energy_xq(x, h)
                dedx = self.energy_jac_x(x, h)
                delta_e = np.dot(-self.Grad, e)

                dmdx = dfdx / delta_e[:, None] - f[:, None] * np.dot(-self.Grad, dedx) / delta_e[:, None] ** 2

            elif "f" in self.optimization_variable and "x" not in self.optimization_variable:
                x = self.x_f(f)
                e = self.energy_xq(x, h)
                dedx = self.energy_jac_x(x, h)
                # noinspection PyUnboundLocalVariable
                dedf = np.dot(dedx, dxdf)
                delta_e = np.dot(-self.Grad, e)

                dmdf = np.identity(self.N) / delta_e - f[:, None] * np.dot(-self.Grad, dedf) / delta_e[:, None] ** 2

            elif "x" in self.optimization_variable and "f" in self.optimization_variable:
                e = self.energy_xq(x, h)
                dedx = self.energy_jac_x(x, h)
                delta_e = np.dot(-self.Grad, e)

                dmdf = np.identity(self.N) / delta_e
                dmdx = - f[:, None] * np.dot(-self.Grad, dedx) / delta_e[:, None] ** 2

            # noinspection PyUnboundLocalVariable
            m = np.divide(f, delta_e)

        q = h * self.qs(x, self.pres)
        delta_q = np.dot(-self.Grad, q)
        dTdx = -cst.Tref / np.power(x, 2)
        dqdT = phy.drsat_dT_vec(self.pres, cst.Tref / x)
        dqdx = np.diag(dqdT * dTdx * h)

        jac_x = np.zeros((self.n + 1, self.x_int * (self.n + 1)))
        jac_z = np.zeros((self.n + 1, self.m_int * self.N))
        jac_c = np.zeros((self.n + 1, self.c_int * self.mu))
        jac_f = np.zeros((self.n + 1, self.f_int * self.N))
        jac_h = np.zeros((self.n + 1, self.h_int * self.n + 1))
        if "x" in self.optimization_variable:
            # noinspection PyUnboundLocalVariable
            jac_x = np.dot(self.Conv, dmdx * delta_q[:, None]) + np.linalg.multi_dot(
                [self.Conv, m[:, None] * -self.Grad, dqdx]) + np.dot(self.Conv, self.transport_jac_x(c, dqdx))
        if "f" in self.optimization_variable:
            # noinspection PyUnboundLocalVariable
            jac_f = np.dot(self.Conv, dmdf * delta_q[:, None]) + np.linalg.multi_dot([self.Conv,
                                                                                      m[:, None] * -self.Grad, dqdx,
                                                                                      dxdf]) + np.dot(self.Conv,
                                                                                                      self.transport_jac_x(
                                                                                                          c,
                                                                                                          np.dot(dqdx,
                                                                                                                 dxdf)))

        if "m" in self.optimization_variable:
            jac_z = np.dot(self.Conv, np.diag(self.dmdz(z) * delta_q))
        if "c" in self.optimization_variable:
            jac_c = np.dot(self.Conv, self.transport_jac_a(c, q))
        if "h" in self.optimization_variable:
            jac_h = np.linalg.multi_dot([self.Conv, m[:, None] * -self.Grad, np.diag(self.qs(x, self.pres))]) \
                    + np.dot(self.Conv, self.transport_jac_x(c, np.diag(self.qs(x, self.pres))))

        jac[:, :self.x_int * (self.n + 1)] = jac_x
        jac[:, self.x_int * (self.n + 1):self.x_int * (self.n + 1) + self.m_int * self.N] = jac_z
        jac[:, self.x_int * (self.n + 1) + self.m_int * self.N:
               self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu] = jac_c
        jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu:
               self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N] = jac_f
        jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N:
               self.x_int * (
                       self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N + self.h_int * (
                       self.n + 1)] = jac_h

        return jac[1:, :] * cst.Lref

    def con_pos_p_error(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain x or f.
        :return: 1D ndarray, the "error" on the constraint on precipitation, computed as max(-precip, 0)_i/mean(|precip|)
        """
        p = self.con_pos_p(v)

        p_err = np.maximum(-p, np.zeros(p.shape)) / np.mean(np.abs(p))

        return p_err

    def con_possibility_precipitation(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain x or f. And it must contain h.
        :return: 1D ndarray of dim=(dim(x) - 1), the constraint P * (h - 1) = 0.
        """
        _, _, _, _, h = self.extract_variable(v)

        p = self.con_pos_p(v)
        con = p * (h[1:] - 1)

        return con

    def con_possibility_precipitation_jac(self, v: np.ndarray) -> np.ndarray:
        """
        :param v: 1D ndarray, variable of the optimization problem, concatenation of (x, z(m), c, f, and/or h)
                  It must contain x or f. And it must contain h.
        :return: 2D ndarray of dim=(dim(x) - 1, len(v)), the jacobian of the constraint P * (h - 1) = 0.
        """
        _, _, _, _, h = self.extract_variable(v)

        p = self.con_pos_p(v)

        jac = self.con_pos_p_jac(v) * (h[1:, None] - 1)
        jac[:, self.x_int * (self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N:
               self.x_int * (
                       self.n + 1) + self.m_int * self.N + self.c_int * self.mu + self.f_int * self.N + self.h_int * (
                       self.n + 1)] += \
            p[:, None] * np.identity(self.n + 1)[1:, :]

        return jac

    def transport(self, c: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Similar to F = -m Grad(y) but for circular convection instead.

        :param c: 1D ndarray of dim mu (a = Cycl * c)
        :param y: 1D ndarray of dim (n+1)
        :return: transport(a, y) = 0.5 * (a (T + S) y - |a| Grad(y))
        """
        if "c" not in self.optimization_variable:
            return np.zeros(self.N)
        D_sum = self.startMat + self.targetMat
        a = np.dot(self.Cycl, c)

        return 0.5 * (a * np.dot(D_sum, y) + np.abs(a) * np.dot(-self.Grad, y))

    def transport_jac_x(self, c: np.ndarray, dydx: np.ndarray) -> np.ndarray:
        """
        Jacobian of transport thanks to "x" variable.

        :param c: 1D ndarray of dim mu (a = Cycl * c)
        :param dydx: 2D ndarray of dim (n+1, n+1)

        :return: d transport(c, y(x)) / dx  (2D ndarray of dim (N, n+1))
        """
        if "c" not in self.optimization_variable:
            return np.zeros((self.N, np.shape(dydx)[1]))
        D_sum = self.startMat + self.targetMat
        # a_t =  np.transpose(a)
        a = np.dot(self.Cycl, c)
        # a_t = np.array([[a[i]] for i in range(N)])
        # print(a_t)
        # print(np.dot(D_sum, dedx))
        # print(a_t*np.dot(D_sum, dedx))
        # print(abs(a_t)*np.dot(-self.Grad,dedx))
        # print("diff jac x", 0.5 * (a_t * np.dot(D_sum, dydx) + np.abs(a_t) * np.dot(-self.Grad, dydx)) - \
        #              0.5 * (a[:, None] * np.dot(D_sum, dydx) + np.abs(a[:, None]) * np.dot(-self.Grad, dydx)))

        return 0.5 * (a[:, None] * np.dot(D_sum, dydx) + np.abs(a[:, None]) * np.dot(-self.Grad, dydx))

    def transport_jac_a(self, c: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Jacobian of transport thanks to "c" variable.

        :param c: 1D ndarray of dim mu (a = Cycl * c)
        :param y: 1D ndarray of dim (n+1)

        :return: d transport(c, y) / dc  (2D ndarray of dim (N, mu))
        """
        D_sum = self.startMat + self.targetMat
        a = np.dot(self.Cycl, c)
        dyda = 0.5 * (np.diag(np.dot(D_sum, y)) + np.sign(a) * np.diag(np.dot(-self.Grad, y)))
        jac = np.dot(dyda, self.Cycl)  # Cycl = dadc
        return jac

    def calcul_physical_variables(self, v: np.ndarray) -> dict:  # compute the physical variables T, e, m, F, q, h, l...
        """
        Calculate all the variables that can be saved and used for plots.

        :param v: 1D ndarray, the concatenation of optimization variables
        :return: dict[str, value], where value is v, T, E, M, A, F, q, h ...
        """
        x, z, c, f, h = self.extract_variable(v)

        # Creation of the dictionary
        results_temp = {'v': v}

        if "f" in self.optimization_variable and "x" not in self.optimization_variable:
            x = self.x_f(f)
        rf = self.radiative_flux(x)
        e = self.energy_xq(x, h)

        if "f" not in self.optimization_variable:
            f = np.dot(self.Trinf, rf)
        if "m" not in self.optimization_variable and "c" not in self.optimization_variable:
            delta_e = np.dot(-self.Grad, e)
            m = np.divide(f, delta_e)
            z = self.z_m(m)
        if "c" in self.optimization_variable:
            a = np.dot(self.Cycl, c)
        if "m" in self.optimization_variable:
            m = self.m_z(z)
        if "h" not in self.optimization_variable:
            h = np.ones(self.n + 1)
        q = h * self.qs(x, self.pres)

        # distinguishing E dry and E wet, F dry and F wet
        e_dry = self.energy_dry(x)
        e_wet = self.energy_wet(x, h)
        E_dry = cst.Eref * e_dry
        E_wet = cst.Eref * e_wet

        f_dry = 0
        f_wet = 0
        if "m" in self.optimization_variable or "c" not in self.optimization_variable:
            # noinspection PyUnboundLocalVariable
            f_dry += m * np.dot(-self.Grad, e_dry)
            f_wet += m * np.dot(-self.Grad, e_wet)
        if "c" in self.optimization_variable:
            f_dry += self.transport(c, e_dry)
            f_wet += self.transport(c, e_wet)

        F_dry = cst.Eref * f_dry
        F_wet = cst.Eref * f_wet

        results_temp['E_dry'], results_temp['E_wet'], \
            results_temp['F_dry'], results_temp['F_wet'] = \
            E_dry, E_wet, F_dry, F_wet

        # l, m
        p = self.con_pos_p(v)

        # Getting the right dimension
        T = cst.Tref / x
        F = cst.Eref * f
        E = cst.Eref * e
        if "c" in self.optimization_variable:
            # noinspection PyUnboundLocalVariable
            results_temp['A'] = a
        if "m" in self.optimization_variable or "c" not in self.optimization_variable:
            results_temp['M'] = m
            results_temp['z'] = z

        theta = T * (cst.p0 * np.ones(self.n + 1) / self.pres) ** (cst.R / cst.Cp)

        # Implementing the results
        results_temp['T'], results_temp['E'], results_temp['q'], \
            results_temp['h'], results_temp['F'], results_temp['P'], \
            results_temp['theta'] = T, E, q, h, F, p, theta

        return results_temp

    def createGraph(self) -> dict:
        """
        dim(startMat) = dim(targetMat) = dim(Grad) = (N, n+1)
        dim(Conv) = (n+1, N)
        dim(Trinf) = (N, n+1)
        dim(Cycl) = (N, N - n)

        For a linegraph:
        startMat =
              [1  0  0  0  0 ..]
              [0  1  0  0  0 ..]
              [0  0  1  0  0 ..]
              ................
              [...   0  1  0  0]
              [...   0  0  1  0]

         targetMat =
              [0  1  0  0  0 ..]
              [0  0  1  0  0 ..]
              [0  0  0  1  0 ..]
              ................
              [...   0  0  1  0]
              [...   0  0  0  1]

         Grad =
              [-1 1  0  0  0 ..]
              [0 -1  1  0  0 ..]
              [0  0 -1  1  0 ..]
              ................
              [...   0 -1  1  0]
              [...   0  0 -1  1]

         Conv =
              [-1 0  0  0  0 ..]
              [1 -1  0  0  0 ..]
              [0  1 -1  0  0 ..]
              [0  0  1 -1  0 ..]
                ................
              [...   0  1 -1  0]
              [...   0  0  1 -1]
              [...   0  0  0  1]

         Trinf =
              [1  0  0  0  0 ..]
              [1  1  0  0  0 ..]
              [1  1  1  1  0 ..]
                ................
              [...   1  1  0  0]
              [...   1  1  1  0]

        Trinf is the pseudo inverse matrix of Div = -Conv.
        We only have the weak relations Trinf = Trinf * Div * Trinf
                                        Div = Div * Trinf * Div

        :return: dict[Grad, Conv, Trinf, N, startMat, targetMat, Cycl]

        """
        N = self.vertices_to_edges()  # the number of edges

        if self.parameters['graph'] == 'linegraph':
            start, target, Cycl = self.linegraph()
            startMat, targetMat = self.make_grad_conv_mat(start, target)

            Grad = targetMat - startMat
            Conv = np.transpose(Grad)

            # #Initial method of Karine Watrin
            Trinf = np.zeros((self.n, self.n + 1))
            for i in range(self.n):
                for j in range(i + 1):
                    Trinf[i, j] = 1

            # # Trinf calculation, in a different way (should be equivalent).
            # Trinf = - np.linalg.pinv(Conv)
            # epsilon = 1e-15
            # Trinf[abs(Trinf) < epsilon] = 0 # get rid of very small absurd numerical values

        elif self.parameters['graph'] == 'stargraph':  # not used in the Q.Pikeroen's master thesis (every vertex are linked).
            start, target, Cycl = self.stargraph(N)
            startMat, targetMat = self.make_grad_conv_mat(start, target)

            Grad = targetMat - startMat
            Conv = np.transpose(Grad)

            # Trinf doesn't work
            # Trinf calculation
            Trinf = - np.linalg.pinv(Conv)
            epsilon = 1e-15
            Trinf[abs(Trinf) < epsilon] = 0  # get rid of very small absurd numerical values

        elif self.parameters['graph'] == 'doublestargraph':  # every vertex but the 0 are linked. The 0 is linked to the 1.
            start, target, Cycl = self.doublestargraph(N)
            startMat, targetMat = self.make_grad_conv_mat(start, target)

            Grad = targetMat - startMat
            Conv = np.transpose(Grad)

            # #Trinf doesn't work
            # # Trinf calculation
            Trinf = - np.linalg.pinv(Conv)
            epsilon = 1e-15
            Trinf[abs(Trinf) < epsilon] = 0  # get rid of very small absurd numerical values

        else:
            print('The graph', self.parameters['graph'], ' hasn\'t been coded yet.')

        # print(np.allclose(-Conv, np.dot(-Conv, np.dot(Trinf, -Conv))))
        # print(np.allclose(Trinf, np.dot(Trinf, np.dot(-Conv, Trinf))))

        # noinspection PyUnboundLocalVariable
        return {"Grad": Grad, "Conv": Conv, "Trinf": Trinf, "N": N,
                "startMat": startMat, "targetMat": targetMat, "Cycl": Cycl}

    @staticmethod
    def set_link(eIndex: int, vStart: int, vEnd: int, start: np.ndarray, target: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:
        """
        Makes a link (edge) between vertices vStart and vEnd.

        :param eIndex: index of the vector (or edge or link).
        :param vStart: index of the box (or vertex) where the vector starts
        :param vEnd: index of the box (or vertex) where the vector ends
        :param start: start(k) corresponds to the vector k and its value is the box number where the vector starts
        :param target: target(k) corresponds to the vector k and its value is the box number where the vector ends

        :return: start, target  (vectors of dim N)
        """
        start[eIndex] = vStart
        target[eIndex] = vEnd

        return start, target

    def make_grad_conv_mat(self, start: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :param start: start(k) corresponds to the vector k and its value is the box number where the vector starts
        :param target: target(k) corresponds to the vector k and its value is the box number where the vector ends

        :return: startMat, targetMat. Matrices of dim (N, n+1), allowing to go from the vertices space to the
                                      edges space.
        """
        m = np.size(start)  # = np.size(target)
        startMat = np.zeros((m, self.n + 1))
        targetMat = np.zeros((m, self.n + 1))
        for k in range(0, m):
            startMat[k, int(start[k])] = 1
            targetMat[k, int(target[k])] = 1
        return startMat, targetMat

    def linegraph(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        n vertex on a line: (1) --1-- (2) --2-- .... --n-1-- (n)

        start(k) corresponds to the vector k and its value is the box number where the vector starts
        target(k) corresponds to the vector k and its value is the box number where the vector ends
        Cycl = 0

        :return: start, target, Cycl
        """

        start = np.zeros(self.n)
        target = np.zeros(self.n)
        for k in range(1, self.n + 1):
            eIndex = k - 1
            start, target = self.set_link(eIndex, k - 1, k, start, target)
        Cycl = 0
        return start, target, Cycl

    def stargraph(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (We prefer using "doublestargraph")

        (1) --1-- (2) --4-- (3) --6-- (4)
        (1)       --2--     (3)
        (1)            --3--          (4)
                  (2)      --5--      (4)

        Cycl is the matrix of elementary circles. For example, the edges (vectors) --1--, --2-- and --4-- form a circle;
        and also --2--, --3-- and --6--; and also --4--, --5-- and --6.

        Cycl = [1  0  0
               -1  1  0
                0 -1  0
                1  0  1
                0  0 -1
                0  1  1]

        start(k) corresponds to the vector k and its value is the box number where the vector starts
        target(k) corresponds to the vector k and its value is the box number where the vector ends

        :return: start, target, Cycl
        """
        kl = 0
        start = np.zeros(N)
        target = np.zeros(N)
        for i in range(0, self.n):
            for j in range(i + 1, self.n + 1):
                start, target = self.set_link(kl, i, j, start, target)
                kl = kl + 1

        # Cycl is not used for now (should be usefull when resolving with x,
        # in order to get (Conv-1) and then F = (Conv)^-1*R(x) )
        Cycl = np.zeros((N, N - self.n))  # N - n is the number of elementary cycles
        kl = 0
        for i in range(self.n - 1):
            for j in range(self.n - i - 1):
                Cycl[j + self.alpha(i), kl] = 1
                Cycl[self.alpha(1 + i + j), kl] = 1
                Cycl[1 + j + self.alpha(i), kl] = -1
                kl = kl + 1

        return start, target, Cycl

    def doublestargraph(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Similar to "stargraph" but the first box is connected to only the second (because physically, the first is
        infinitely small).

        (1) --1-- (2) --2-- (3) --4-- (4)
                  (2)      --3--      (4)

        Cycl is the matrix of elementary circles. For example, the edges (vectors) --2--, --3-- and --4-- form a circle.

        Cycl = [0
                1
               -1
                1]

        start(k) corresponds to the vector k and its value is the box number where the vector starts
        target(k) corresponds to the vector k and its value is the box number where the vector ends

        :return: start, target, Cycl
        """
        start = np.zeros(N)
        target = np.zeros(N)
        start[0] = 0
        target[0] = 1
        kl = 1
        for i in range(1, self.n):
            for j in range(i + 1, self.n + 1):
                # print(i,j,kl)
                start, target = self.set_link(kl, i, j, start, target)
                kl = kl + 1

        Cycl = np.zeros((N, N - self.n))  # N - n is the number of elementary cycles
        # print(Cycl)
        kl = 0
        for i in range(1, self.n - 1):
            for j in range(0, self.n - i - 1):
                # print(i,j)
                Cycl[j + self.alpha_2(i), kl] = 1
                Cycl[self.alpha_2(1 + i + j), kl] = 1
                Cycl[1 + j + self.alpha_2(i), kl] = -1
                kl = kl + 1

        return start, target, Cycl

    def alpha(self, k: int) -> int:
        """
        For stargraph.
        """
        return int((self.n + 1) * k - k * (k + 1) / 2)

    def alpha_2(self, k: int) -> int:
        """
        For doublestargraph.
        """
        return 1 + int(self.n * (k - 1) - k * (k - 1) / 2)

    def vertices_to_edges(self) -> int:
        """
        :return: the number N of edges (or links or vectors).
        """
        n = self.parameters['number of levels']
        if self.parameters['graph'] == 'linegraph':
            N = n
        elif self.parameters['graph'] == 'stargraph':
            N = int(n * (n + 1) / 2)
        elif self.parameters['graph'] == 'doublestargraph':
            N = int(n * (n - 1) / 2) + 1
        # print('N=',N,'n=',n)

        # noinspection PyUnboundLocalVariable
        return N

    def create_initial_value(self, dictionary: dict, str_values: str) -> list[np.ndarray]:
        """
        Take a result, and create initial conditions from it. Useful to run the optimization more than once with
        different parameters.

        :param dictionary: results for an instant, dict[str, value]
        :param str_values: str, concatenation of optimization variables to initialize
        :return: a list of ndarray, that can be used as the value of parameters["initial value"]
        """
        initial_value = []
        if "x" in str_values:
            initial_value.append(cst.Tref / dictionary["T"])
        if "m" in str_values:
            initial_value.append(dictionary["M"])
        if "c" in str_values:
            initial_value.append(np.dot(np.linalg.pinv(self.Cycl), dictionary["A"]))
        if "f" in str_values:
            initial_value.append(1 / cst.Eref * dictionary["F"])
        if "h" in str_values:
            initial_value.append(dictionary["h"])
        return initial_value

    def test(self, v):
        """
        To be completed by user.
        """
        return None
