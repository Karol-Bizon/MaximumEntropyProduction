# -*- coding: utf-8 -*-
"""
MEP Optimization. Here is everything built around the call of the minimize function.
"""

import numpy as np
from numpy import ndarray

# import radiatif as rad
from mep_physics import MepPhysics
import constants as cst
from scipy.optimize import minimize
from scipy.optimize import optimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
import time
from typing import Callable  # Union, Optional, Tuple, List


# from pyswarm import pso  # 'pip install --upgrade pyswarm' in console (linux)


class Optimization:
    """
    Run the optimization process one or several times (mainly scipy.optimize.minimize) with the adequate
    objective function, constraints, jacobians and boundaries,
    """

    def __init__(self, parameters):
        """
        :param parameters: dict[str, Any]
        """
        self.parameters = dict(parameters)
        self.print_option = parameters['print option']

        self.mep_phy = MepPhysics(self.parameters)

        self.n = self.parameters["number of levels"]
        self.N = self.mep_phy.N
        self.mu = self.N - self.n

        self.physical_model = self.parameters['physical model']
        self.optimization_method = self.parameters['optimization method']
        self.optimization_variable = self.parameters['optimization variable']

        self.ofs = -1  # objective function sign
        if 'test' in self.parameters['feasibility']:
            self.feasibility_test = True
            if self.parameters['feasibility objective function'] in ['sum equal', 'minimal sum']:
                self.ofs = 1
        else:
            self.feasibility_test = False

    def variable_initialisation(self) -> np.ndarray:
        """
        :params: None
        :variable_to_initialize: string of names of variables (x, m, z(m), c, f and/or h) to initialize.
                             It can be different from optimization_variable. Default values are taken when one is missing.
        :initial_value: [float or ndarray, ...] list of initial values for each variable to initialize
        :return: 1D ndarray containing all initial conditions side by side in the order (x, z(m), c, f, h)
        """

        # default initial conditions
        x0 = np.ones(self.n + 1)
        z0 = 0.5 * np.ones(self.N)
        c0 = 1e-4 * np.ones(self.mu)
        f0 = np.ones(self.N)
        h0 = 0.5 * np.ones(self.n + 1)

        if 'optimization variable to initialize' in self.parameters:
            variable_to_initialize = self.parameters['optimization variable to initialize']
            initial_value = self.parameters['initial value']

            # position of variables (if not present, rfind=-1)
            pos_x_ini = variable_to_initialize.rfind("x")
            pos_m_ini = variable_to_initialize.rfind("m")
            pos_z_ini = variable_to_initialize.rfind("z")
            pos_c_ini = variable_to_initialize.rfind("c")
            pos_f_ini = variable_to_initialize.rfind("f")
            pos_h_ini = variable_to_initialize.rfind("h")

            # extract the initial condition for each variable
            if pos_x_ini >= 0:
                if type(initial_value[pos_x_ini]) == np.ndarray:
                    x0 = np.array(initial_value[pos_x_ini])
                else:
                    x0 = initial_value[pos_x_ini] * np.ones(self.n + 1)
            if pos_m_ini >= 0:
                if type(initial_value[pos_m_ini]) == np.ndarray:
                    m0 = np.array(initial_value[pos_m_ini])
                else:
                    m0 = initial_value[pos_m_ini] * np.ones(self.N)
                z0 = self.mep_phy.z_m(m0)
            if pos_z_ini >= 0:
                if type(initial_value[pos_z_ini]) == np.ndarray:
                    z0 = np.array(initial_value[pos_z_ini])
                else:
                    z0 = initial_value[pos_z_ini] * np.ones(self.N)
            if pos_c_ini >= 0:
                if type(initial_value[pos_c_ini]) == np.ndarray:
                    c0 = np.array(initial_value[pos_c_ini])
                else:
                    c0 = initial_value[pos_c_ini] * np.ones(self.mu)
            if pos_f_ini >= 0:
                if type(initial_value[pos_f_ini]) == np.ndarray:
                    f0 = np.array(initial_value[pos_f_ini])
                else:
                    f0 = initial_value[pos_f_ini] * np.ones(self.N)
            if pos_h_ini >= 0:
                if type(initial_value[pos_h_ini]) == np.ndarray:
                    h0 = np.array(initial_value[pos_h_ini])
                else:
                    h0 = initial_value[pos_h_ini] * np.ones(self.n + 1)

        v0 = np.empty(0)

        if "x" in self.optimization_variable:
            v0 = np.array(x0)
        if "m" in self.optimization_variable:
            v0 = np.concatenate((v0, z0))
        if "c" in self.optimization_variable:
            v0 = np.concatenate((v0, c0))
        if "f" in self.optimization_variable:
            v0 = np.concatenate((v0, f0))
        if "h" in self.optimization_variable:
            v0 = np.concatenate((v0, h0))

        # linearisation initialisation
        self.mep_phy.param_radiative_flux(x0)  # creates r and r0

        print("Initial Conditions = ", v0)

        return v0

    def objective_function_choice(self) -> tuple[list[Callable[[np.ndarray], float]], Callable[[np.ndarray], float]]:
        """
        Definition of the objective function, depending on the value of feasibilitytest.

        :params: None
        :return: tuple (list_objective, entropy_prod)
                 list_objective = [objective_function, objective_jac, objective_hess]
                 entropy_prod will be used as a constraint (entropy > 0) if parameters["positive entropy production"] = "Yes"
        """
        objective_function, objective_jac, objective_hess, entropyprod = None, None, None, None
        list_objective = []

        # If "entropy variable" isn't specified or wrongly specified, it is taken to its default value, "x".
        if "entropy variable" in self.parameters.keys():
            entropy_variable = self.parameters["entropy variable"]
        else:
            entropy_variable = "x"

        if "entropy variable" in self.parameters.keys() and "x" in self.optimization_variable and \
                not all([char in self.optimization_variable for char in entropy_variable]):
            print("WARNING, all the entropy variables are not in the optimization variables\n"
                  "         We change \"entropy variable\" to \"entropy variable\" = \"x\"")
            entropy_variable = "x"

        # Choice of the objective function, jac and hess depending on the problem
        if self.feasibility_test:
            objective_function = self.mep_phy.objective_function_feasibility
            objective_jac = self.mep_phy.objective_function_feasibility_jac

        elif self.optimization_method != "matrix":

            if "x" in self.optimization_variable:
                if entropy_variable == "x":
                    objective_function = self.mep_phy.minus_entropyprod_x
                    objective_jac = self.mep_phy.minus_entropyprod_jac_x
                    objective_hess = self.mep_phy.minus_entropyprod_hess_x
                elif entropy_variable == "xm":
                    objective_function = self.mep_phy.minus_entropyprod_xz
                    objective_jac = self.mep_phy.minus_entropyprod_jac_xz
                elif entropy_variable == "xf":
                    objective_function = self.mep_phy.minus_entropyprod_xf
                    objective_jac = self.mep_phy.minus_entropyprod_jac_xf
            elif "f" in self.optimization_variable:
                objective_function = self.mep_phy.minus_entropyprod_f
                objective_jac = self.mep_phy.minus_entropyprod_jac_f

        def entropyprod(v):
            return - objective_function(v)

        # Creation of the list containing the objective function, jac and hess if necessary

        if objective_function is not None:
            list_objective.append(objective_function)
            if objective_jac is not None:
                list_objective.append(objective_jac)
                if objective_hess is not None:
                    list_objective.append(objective_hess)
        elif self.optimization_method != "matrix":
            print('The entropy has not been calculated for this optimization variable or entropy variable')

        print("list objective", [objective.__name__ for objective in list_objective])
        return list_objective, entropyprod

    #
    def constraint_choice(self, entropyprod: Callable[[np.ndarray], float] | None) -> \
            tuple[list[ndarray], list[list[Callable]], list[list[Callable]], list[Callable]]:
        """
        Definition of the constraints, depending on the physical model and the optimization variable.

        :param: entropyprod, function that is used as a constraint (entropyprod > 0) if specified in parameters.
        :return: tuple list_bound, list_eq, list_ineq, list_err.
        list_bound contains min and max possible values of variables in the optimization variables,
        list_eq and list_ineq contain equality or inequality constraints for the optimization problem,
        list_err contains a list of functions estimating the error made, but is not used during the optimization process.
        """

        list_eq = []
        list_ineq = []
        list_err = []

        # bounds
        z_min = []
        z_max = []
        if "x" in self.optimization_variable:
            z_min.extend([0.5] * (self.n + 1))
            z_max.extend([2] * (self.n + 1))
        if "m" in self.optimization_variable:
            z_min.extend([0] * self.N)
            z_max.extend([self.mep_phy.z_max] * self.N)
        if "c" in self.optimization_variable:
            z_min.extend([-np.inf] * self.mu)
            z_max.extend([np.inf] * self.mu)
        if "f" in self.optimization_variable:
            z_min.extend([-np.inf] * self.N)
            z_max.extend([np.inf] * self.N)
        if "h" in self.optimization_variable:
            z_min.extend([0] * (self.n + 1))
            z_max.extend([1] * (self.n + 1))
        z_min = np.array(z_min)
        z_max = np.array(z_max)
        list_bound = [z_min, z_max]

        # constraints
        if (self.optimization_variable == "x" or self.optimization_variable == "xh") \
                and self.optimization_method != "matrix":
            list_eq.append([self.mep_phy.con_global_energy_balance, self.mep_phy.con_global_energy_balance_jac])
            list_err.append(self.mep_phy.con_global_energy_balance_error)
            # if self.physical_model == "un" and self.optimization_method != 'matrix':
        if self.physical_model != "un":
            if "m" not in self.optimization_variable and "c" not in self.optimization_variable:
                list_ineq.append(
                    [self.mep_phy.con_pos_alpha, self.mep_phy.con_pos_alpha_jac])  # or self.mep_phy.con_pos_m
                list_err.append(self.mep_phy.con_pos_alpha_error)
            if "x" in self.optimization_variable and (
                    "m" in self.optimization_variable or "c" in self.optimization_variable
                    or "f" in self.optimization_variable):
                list_eq.append([self.mep_phy.con_local_energy_balance, self.mep_phy.con_local_energy_balance_jac])
                list_err.append(self.mep_phy.con_local_energy_balance_error)
            if "f" in self.optimization_variable and (
                    "m" in self.optimization_variable or "c" in self.optimization_variable):
                list_eq.append([self.mep_phy.con_def_convective_flux, self.mep_phy.con_def_convective_flux_jac])
                list_err.append(self.mep_phy.con_def_convective_flux_error)
            if "wc" in self.physical_model:
                list_ineq.append([self.mep_phy.con_pos_p, self.mep_phy.con_pos_p_jac])
                list_err.append(self.mep_phy.con_pos_p_error)

                if "ns" in self.physical_model:
                    list_eq.append(
                        [self.mep_phy.con_possibility_precipitation, self.mep_phy.con_possibility_precipitation_jac])

        if 'positive entropy production' in self.parameters and self.parameters[
           'positive entropy production'] == 'Yes' and self.optimization_method != 'matrix':
            list_ineq.append([entropyprod])

        if self.optimization_method != "matrix":
            print("list bound", list_bound)
            print("list_eq", [eq.__name__ for couple_eq in list_eq for eq in couple_eq])
            print("list ineq", [ineq.__name__ for couple_ineq in list_ineq for ineq in couple_ineq])
            print("list err", [err.__name__ for err in list_err])
            print("entropyprod", [entropyprod.__name__])

        # noinspection PyTypeChecker
        return list_bound, list_eq, list_ineq, list_err

    def constraint_implementation(self, list_eq: list[list[Callable]], list_ineq: list[list[Callable]],
                                  list_bound: list[ndarray]) -> tuple[list, Bounds | None]:
        """
        Put the equality and inequality constraints, and the boundaries, in a syntax understood by the optimization
        algorithm.

        :param list_eq: the list of equality constraints, in the form [[eq_1, jac_1], [eq_2, jac2]...], with jac being
        the jacobian, and is optional.
        :param list_ineq: the list of inequality constraints, in the same form as list_eq.
        :param list_bound: a list of 2 numpy arrays having the dimension of the optimization variable, [v_min, v_max]
        :return list_cons, bound: a tuple with a list of constraints, and a list of bounds.
        """
        # Implementation of the bounds
        list_cons = []
        bound = None
        if list_bound:
            v_min, v_max = list_bound
            bound = Bounds(v_min, v_max)

        # Implementing the constraint
        if self.optimization_method == 'SLSQP':
            for eq in list_eq:
                if len(eq) == 2:
                    list_cons.append({'type': 'eq', 'fun': eq[0], 'jac': eq[1]})
                else:
                    list_cons.append({'type': 'eq', 'fun': eq[0]})
            for ineq in list_ineq:
                if len(ineq) == 2:
                    list_cons.append({'type': 'ineq', 'fun': ineq[0], 'jac': ineq[1]})
                else:
                    list_cons.append({'type': 'ineq', 'fun': ineq[0]})

        if self.optimization_method == 'trust-constr':
            for eq in list_eq:
                if len(eq) == 2:
                    list_cons.append(NonlinearConstraint(eq[0], 0, 0, jac=eq[1]))
                else:
                    list_cons.append(NonlinearConstraint(eq[0], 0, 0))
            for ineq in list_ineq:
                if len(ineq) == 2:
                    list_cons.append(NonlinearConstraint(ineq[0], 0, +np.inf, jac=ineq[1]))
                else:
                    list_cons.append(NonlinearConstraint(ineq[0], 0, +np.inf))
                if len(ineq) > 2:
                    v_min_cons, v_max_cons = ineq[1], ineq[2]
                    if len(ineq) == 3:
                        # noinspection PyTypeChecker
                        list_cons.append(NonlinearConstraint(ineq[0], v_min_cons, v_max_cons))
                    else:
                        # noinspection PyTypeChecker
                        list_cons.append(NonlinearConstraint(ineq[0], v_min_cons, v_max_cons, jac=ineq[3]))
        return list_cons, bound

    @staticmethod
    def constraint_value(v: ndarray, results: dict, list_all_cons: list[list[Callable]],
                         list_err: list[Callable]) -> dict:
        """
        Put the values of the constraints in the final results.

        :param v: 1D np.ndarray, concatenation of all optimization variables
        :param list_all_cons: list_eq + list_ineq
        :param list_err: list of errors associated with list_all_cons
        :param results: dict[str, dict], results[instant][str]
        :return results: dict, with added constraints results
        """
        # Implementation of the value of the constraint on the v point in the final results dictionary
        for i in range(len(list_all_cons)):
            cons_fun = list_all_cons[i][0]
            cons_name = cons_fun.__name__
            value = cons_fun(v)

            # Put the constraint in the results
            # but not for the entropyprod which is already put elsewhere
            if 'entropyprod' not in cons_name:
                results['final'][cons_name] = value

        for i in range(len(list_err)):
            cons_fun_err = list_err[i]
            cons_name_err = cons_fun_err.__name__
            value_err = cons_fun_err(v)
            results['final'][cons_name_err] = value_err

        return results

    def resolution_main(self) -> dict:
        """
        Essentially, this function creates the initialization, the constraints, the boundaries, run the
        optimization and save the results.
        :return results: a dictionary, results[instant][value]
        """
        # Implementing the measure of time
        start_time_resolution = time.time()
        # Printing the beginning of the resolution
        model_name = self.parameters['model name']
        if self.feasibility_test:
            print('\n', '\n', '----- Resolution of the feasibility test -----', '\n', '----- for the', model_name,
                  'problem -----', '\n')
        else:
            print('\n', '\n', '----- Resolution of the problem -----', '\n', '----- for the', model_name,
                  'problem -----',
                  '\n')

        # Initialisation of the optimization variables
        v0 = self.variable_initialisation()
        results = {}

        if self.parameters['initial value save option'] or self.parameters['initial value plot option']:
            results['initial'] = self.mep_phy.calcul_physical_variables(v0)
            # calcul of entropy
            x, _, _, f, _ = self.mep_phy.extract_variable(v0)
            if "f" in self.optimization_variable and "x" not in self.optimization_variable:
                x = self.mep_phy.x_f(f)

            results['initial']['entropy'] = cst.Sover4 / cst.Tref * self.mep_phy.minus_entropyprod_x(x)

        # Initialization of the objective function
        list_objective, entropyprod = self.objective_function_choice()

        # Declaration of the constraint
        list_bound, list_eq, list_ineq, list_err = self.constraint_choice(entropyprod)
        list_all_cons = list_eq + list_ineq
        list_cons, bound = self.constraint_implementation(list_eq, list_ineq, list_bound)

        # Resolution of the problem, with the resolution method chosen
        v, res, nb_iteration, results = self.resolution_iterative_core(v0, results, list_objective, list_cons, bound,
                                                                       list_bound)

        x, z, _, f, _ = self.mep_phy.extract_variable(v)
        if "f" in self.optimization_variable and "x" not in self.optimization_variable:
            x = self.mep_phy.x_f(f)

        # Implementation of the final results
        results['final'] = self.mep_phy.calcul_physical_variables(v)
        results['final']['convergence'] = True if res.success else False
        results["final"]["number of iteration"] = str(nb_iteration) + '/' + str(
            self.parameters['max number of iterations'])
        results = self.constraint_value(v, results, list_all_cons, list_err)

        if self.feasibility_test:
            results['final']['entropy'] = cst.Sover4 / cst.Tref * self.ofs * self.mep_phy.minus_entropyprod_x(x)
            results["final"]["feasibility objective function"] = self.ofs * res.fun
            if res.success:
                print('The problem is feasible')
            else:
                print('The problem seems unfeasible for the chosen parameters')
        else:
            results['final']['entropy'] = cst.Sover4 / cst.Tref * self.ofs * res.fun

        if self.print_option:
            print('Final results for the ', model_name, ' problem :')
            for value in results['final'].keys():
                print(value, ' :', results['final'][value])
            print('Number of iteration :', nb_iteration)
            print('Duration : ', time.time() - start_time_resolution, '\n', '\n')

        return results

    def resolution_iterative_core(self, v: ndarray, results: dict, list_objective: list, list_cons: list, bound: Bounds,
                                  list_bound: list[ndarray]) -> tuple[np.ndarray, optimize.OptimizeResult, int, dict]:
        """
        :param v: concatenation of optimization variables
        :param results: dict of results
        :param list_objective: [objective function, its jacobian]
        :param list_cons: list of constraints
        :param bound: Bounds instance
        :param list_bound: [v_min, v_max]
        :return: tuple (v, res (resolution), iteration, results)

        """

        # Initialisation of the option evolution
        iteration_save_plot_option = self.parameters['evolution save option'] or self.parameters[
            "evolution plot option"]
        if iteration_save_plot_option:
            nb_iteration_for_saving_plotting = self.parameters['nb iteration for saving and plotting']

        # Sub-function to print and save the intermediary results
        def intermediary_result():

            intermediary = 'int.' + str(iteration)
            results[intermediary] = self.mep_phy.calcul_physical_variables(v)
            results[intermediary]['entropy'] = cst.Sover4 / cst.Tref * self.ofs * res.fun

            if self.print_option:
                print('Intermediary results for the iteration ', iteration)
                for value in results[intermediary].keys():
                    print(value, ' = ', results[intermediary][value])
                print('\n')

        # Matrix problem resolution (for unconstrained)
        if self.optimization_method == 'matrix':
            if self.optimization_variable != "x":
                raise Exception("The optimization variable should be \"x\" when optimization method is matrix.")
            if self.physical_model != "un":
                raise Exception("The physical model should be \"un\" when optimization method is matrix.")
            # Initialisation
            mepStep = self.mepStepuncon
            nb_iteration = self.parameters['max number of iterations']

            # Iterative resolution
            for iteration in range(1, nb_iteration + 1):

                v, res = mepStep(v, list_objective, list_cons, bound, list_bound)
                # res.nit = iteration # useless I guess

                if self.parameters['evolution save option'] or self.parameters["evolution plot option"]:
                    intermediary = 'int.' + str(iteration)
                    results[intermediary] = self.mep_phy.calcul_physical_variables(v)
                    print('Iteration ', iteration)
                    print('T :', results[intermediary]['T'])
                    print('E :', results[intermediary]['E'], '\n')

        # Scipy problem resolution
        else:

            # Resolving for the different resolution method
            resolution_method = self.parameters['resolution method']
            mepStep = self.mepStepcon

            # Simple method
            if resolution_method == 'simple':

                nb_iteration = self.parameters['max number of iterations']

                for iteration in range(1, nb_iteration + 1):

                    if self.print_option:
                        print('Iteration', iteration, 'of resolution')

                    (v, res) = mepStep(v, list_objective, list_cons, bound, list_bound)

                    if iteration_save_plot_option:
                        # noinspection PyUnboundLocalVariable
                        if iteration % nb_iteration_for_saving_plotting == 0:
                            intermediary_result()

            # Minimum entropy variation threshold method
            if resolution_method == 'threshold':

                threshold, nb_iteration_max = self.parameters['variation threshold'], self.parameters[
                    'max number of iterations']
                iteration, nb_iteration_after_conv = 0, -1

                while (iteration < nb_iteration_max) and (nb_iteration_after_conv < 3):
                    iteration += 1

                    if self.print_option:
                        print('Iteration', iteration, 'of resolution')

                    (v, res) = mepStep(v, list_objective, list_cons, bound, list_bound)

                    if iteration_save_plot_option:
                        if iteration % nb_iteration_for_saving_plotting == 0:
                            intermediary_result()

                    if not res.success:
                        nb_iteration_after_conv = -1

                    if res.success:
                        if nb_iteration_after_conv == -1:
                            entropy_old = self.ofs * res.fun
                            v_0_old = v[0]
                            nb_iteration_after_conv = 0
                        else:
                            entropy = self.ofs * res.fun
                            # noinspection PyUnboundLocalVariable
                            entropy_variation = abs(entropy - entropy_old)
                            # noinspection PyUnboundLocalVariable
                            value_variation = abs(v[0] - v_0_old)
                            entropy_old = entropy
                            v_0_old = v[0]
                            nit_conv = res.nit
                            if entropy_variation < threshold and value_variation < threshold and res.nit == nit_conv:
                                nb_iteration_after_conv += 1
                            else:
                                nb_iteration_after_conv = 0

            # Convergence of the mepStepcon function method
            if resolution_method == 'convergence':  # It only takes into account v[0] !

                nb_iteration_max = self.parameters['max number of iterations']
                iteration, nb_iteration_after_conv, nit_conv = 0, 0, -1

                while (iteration < nb_iteration_max) and (nb_iteration_after_conv < 3):
                    iteration += 1

                    if self.print_option:
                        print('Iteration', iteration, 'of resolution')
                    (v, res) = mepStep(v, list_objective, list_cons, bound, list_bound)

                    if iteration_save_plot_option:
                        if iteration % nb_iteration_for_saving_plotting == 0:
                            intermediary_result()

                    if not res.success:
                        nit_conv = -1

                    else:
                        if nit_conv == -1:
                            nit_conv = res.nit
                            entropy_conv = self.ofs * res.fun
                            v_0_conv = res.x[0]
                        else:
                            # noinspection PyUnboundLocalVariable
                            if nit_conv == res.nit and entropy_conv == self.ofs * res.fun and abs(
                                    v[0] - v_0_conv) < 10 ** -5:
                                nb_iteration_after_conv += 1
                            else:
                                nit_conv, entropy_conv, v_0_conv = res.nit, self.ofs * res.fun, v[0]
                                nb_iteration_after_conv = 0

            # Tabu research method (at each iteration, the entry of mepStepcon is the best result known)
            if resolution_method == 'tabu':

                if self.print_option:
                    print('Iteration', 1, 'of resolution')

                nb_iteration_max = self.parameters['max number of iterations']
                (v_best, res) = mepStep(v, list_objective, list_cons, bound, list_bound)
                entropy_best = self.ofs * res.fun
                iteration, nb_iteration_after_conv = 1, 0

                while iteration < nb_iteration_max and nb_iteration_after_conv < 5:

                    iteration += 1
                    if self.print_option:
                        print('Iteration', iteration, 'of resolution')

                    (v, res) = mepStep(v_best, list_objective, list_cons, bound, list_bound)

                    if iteration_save_plot_option:
                        if iteration % nb_iteration_for_saving_plotting == 0:
                            intermediary_result()

                    if res.nit == 1:
                        nb_iteration_after_conv += 1
                    entropy = self.ofs * res.fun
                    if self.ofs * entropy < self.ofs * entropy_best:
                        entropy_best, v_best = entropy, v

                v, entropy = v_best, entropy_best

            # Variation of the max number or iteration method
            if resolution_method == 'maxiter variation':

                list_maxiter, nb_iteration = self.parameters['list maxiter'], self.parameters['number of iterations']
                nb_maxiter_to_test = len(list_maxiter)
                iteration, compt_0, nit_conv, entropy_conv = 0, 0, 10002, -10 ** 10

                while compt_0 < nb_maxiter_to_test:

                    maxiter = list_maxiter[compt_0]
                    self.parameters['maxiter of the minimize function'] = maxiter
                    compt_0 += 1
                    compt_1, compt_2 = 0, 0

                    # Phase 1 : wait mepStep to be under the maxiter
                    while compt_1 < nb_iteration:
                        compt_1 += 1
                        iteration += 1
                        (v, res) = mepStep(v, list_objective, list_cons, bound, list_bound)

                        if res.success:
                            compt_1 = nb_iteration + 1

                        if iteration_save_plot_option:
                            if iteration % nb_iteration_for_saving_plotting == 0:
                                intermediary_result()

                    if compt_1 == nb_iteration:
                        compt_2 = nb_iteration + 2

                    # Phase 2 : wait mepStep to converge
                    # noinspection PyUnboundLocalVariable
                    if (nit_conv == res.nit) and (self.ofs * res.fun == entropy_conv):
                        compt_2 = nb_iteration + 2

                    compt_3 = 0
                    while compt_2 < nb_iteration:
                        compt_2 += 1
                        iteration += 1

                        (v, res) = mepStep(v, list_objective, list_cons, bound, list_bound)

                        if res.success:
                            if (res.nit == nit_conv) and (self.ofs * res.fun == entropy_conv):
                                compt_3 += 1
                                if compt_3 > 5:
                                    compt_2 = nb_iteration + 2
                            else:
                                compt_3 = 0
                                nit_conv, entropy_conv = res.nit, self.ofs * res.fun
                        else:
                            compt_3 = 0
                            nit_conv, entropy_conv = res.nit, self.ofs * res.fun

                        if iteration_save_plot_option:
                            if iteration % nb_iteration_for_saving_plotting == 0:
                                intermediary_result()

            # Variation of initial values method
            if resolution_method == 'value variation':

                variable = self.parameters['variable to change']
                nb_iteration = self.parameters['number of iterations']
                [v_min, v_max] = self.parameters['min-max of the variable']
                nb_iteration_per_cycle = self.parameters['number of iteration per cycle']

                if variable == 'm':
                    ind_min = self.n + 1
                    ind_max = 2 * self.n + 1
                if variable == 'x':
                    ind_min = 0
                    ind_max = self.n + 1
                if variable == 'f':
                    ind_min = self.n + 1
                    ind_max = 2 * self.n + 1
                # noinspection PyUnboundLocalVariable
                entropy_best, variable_best = -10 ** 100, v[ind_min:ind_max]

                for iteration in range(1, nb_iteration + 1):

                    if self.print_option:
                        print('Iteration ', iteration)

                    if (iteration % nb_iteration_per_cycle) == 0:

                        # Random change
                        ind_change = np.random.randint(ind_min, ind_max)
                        new_value = np.random.random() * (v_max - v_min) + v_min
                        v[ind_change] = new_value
                        if self.print_option:
                            print('new value ', new_value, 'for ', variable, ' index ', ind_change)

                        # Selection of the best results for the next iteration
                        (v, res) = mepStep(v, list_objective, list_cons, bound, list_bound)
                        entropy = self.ofs * res.fun
                        if self.ofs * entropy < self.ofs * entropy_best:
                            variable_best = v[ind_min:ind_max]
                        else:
                            v[ind_min:ind_max] = variable_best

                    else:
                        (v, res) = mepStep(v, list_objective, list_cons, bound, list_bound)
                        entropy_best = self.ofs * res.fun
                    if iteration_save_plot_option:
                        if iteration % nb_iteration_for_saving_plotting == 0:
                            intermediary_result()
        # noinspection PyUnboundLocalVariable
        return v, res, iteration, results

    # Verification of the respect of the bounds and of the constraint
    @staticmethod
    def verification_bounds(v: ndarray, list_bound: list) -> ndarray:
        """
        :param v: 1D ndarray
        :param list_bound: [v_min, v_max]
        :return: 1D ndarray, a new v contained in the boundaries.
        """
        if list_bound:
            v_min, v_max = list_bound[0], list_bound[1]
            for i in range(len(v_min)):
                while v[i] < v_min[i] or v[i] > v_max[i]:
                    if v[i] > v_max[i]:
                        v[i] = 0.90 * v_max[i]
                        print('v[', i, '] is superior to the max')
                    if v[i] < v_min[i]:
                        v[i] = 1.1 * v_min[i]
                        print('v[', i, '] is inferior to the min')
        return v

    # MepStep for unconstrained problem (optimization method = 'matrix')
    def mepStepuncon(self, x0: ndarray, _list_objective: list, _list_cons: list, _bound: Bounds,
                     list_bound: list) -> tuple[ndarray, object]:
        """
        Solves the unconstrained MEP problem assuming a linear radiative code R(x) = r.x + r0 (method == "matrix")

        :param x0: ndarray, x0 = Tref / T
        :param _list_objective: not used
        :param _list_cons: not used
        :param _bound: not used
        :param list_bound: [x_min, x_max]
        :return: sol (ndarray), res (resolution, instance of Optimize)
        """
        # Linearisation of the radiative flux
        self.mep_phy.param_radiative_flux(x0)  # creates global r, r0 in mep_physics
        #   | (r+rT)   -sRT |.|x|=  |-r0
        #   |  -sR       0  | |ß|   |sR0
        res = self.mep_phy.maximum_entropyprod()
        sol = self.verification_bounds(res.x, list_bound)
        # print('entropy=',res.fun)
        return sol, res

    # MepStep for constrained problem (optimization method = 'SLSQP' or 'trust-constr'
    def mepStepcon(self, v: ndarray, list_objective: list, list_cons: list, bound: Bounds, list_bound: list[ndarray]) \
            -> tuple[ndarray, optimize.OptimizeResult]:
        """
        Run the minimize function from scipy.optimize.

        :param v: ndarray
        :param list_objective: [objective function, its jacobian]
        :param list_cons: list of constraints
        :param bound: instance of class Bounds
        :param list_bound: [v_min, v_max]
        :return: tuple (optimization variable, resolution)
        """
        # linearisation of radiative fluxes
        x, _, _, f, _ = self.mep_phy.extract_variable(v)
        if "f" in self.optimization_variable and "x" not in self.optimization_variable:
            x = self.mep_phy.x_f(f)

        self.mep_phy.param_radiative_flux(x)

        # Resolution of the problem with the implementation of the scipy.optimize.minimize algorithm

        objective_function = list_objective[0]
        if len(list_objective) == 1:
            # noinspection PyTypeChecker
            res = minimize(objective_function, v, method=self.optimization_method, constraints=list_cons,
                           bounds=bound,
                           options={'maxiter': self.parameters['maxiter of the minimize function']})
        else:
            objective_jac = list_objective[1]
            if len(list_objective) == 2:
                # noinspection PyTypeChecker
                res = minimize(objective_function, v, method=self.optimization_method, jac=objective_jac,
                               constraints=list_cons,
                               bounds=bound,
                               options={'maxiter': self.parameters['maxiter of the minimize function']})
            else:
                objective_hess = list_objective[2]
                # noinspection PyTypeChecker
                res = minimize(objective_function, v, method=self.optimization_method, jac=objective_jac,
                               hess=objective_hess,
                               constraints=list_cons, bounds=bound,
                               options={'maxiter': self.parameters['maxiter of the minimize function']})

        # Results
        v = res.x
        x, z, c, f, h = self.mep_phy.extract_variable(v)
        if "f" in self.optimization_variable and "x" not in self.optimization_variable:
            x = self.mep_phy.x_f(f)

        if self.print_option:
            if self.feasibility_test:
                print('Feasibility objective function', '=', self.ofs * res.fun)
            else:
                print('Entropy Production', ' = ', cst.Sover4 / cst.Tref * self.ofs * res.fun)
            print('Temperature = ', cst.Tref / x - 273.15)
            if 'm' in self.optimization_variable:
                print('Mass = ', self.mep_phy.m_z(z))
            if "c" in self.optimization_variable:
                print("c =", c)
            if "f" in self.optimization_variable:
                print("F = ", cst.Eref * f)
            if "h" in self.optimization_variable:
                print("h = ", h)

            print('Success : ', res.success)
            if not res.success:
                print('Reason of failure : ', res.message)
            print('Number of iterations  : ', res.nit)
            print('')

        v = self.verification_bounds(v, list_bound)

        return v, res

    def test(self, parameter):
        # Parameters definition
        self.mep_phy = MepPhysics(parameter)
        v = self.variable_initialisation()
        results = self.mep_phy.test(v)

        # results = {}
        return results
