# -*- coding: utf-8 -*-
"""
Intermediate. Shapes the parameters and run the optimization.
"""
import numpy as np
from mep_optimization import Optimization
from mep_physics import MepPhysics
from plot_save import PlotSave
from copy import deepcopy


class Intermediate:
    """
    Run the resolution(s) for one or several parameters dictionaries.
    """

    def __init__(self, parameters: dict[str, str | int | float | list | bool],
                 resolution_choice: dict[str, str | list]):
        """
        :param parameters: dictionary of parameters filled in by user in main.py
        :param resolution_choice: dictionary indicating the choice of resolution
        """
        self.parameters = parameters
        self.list_parameters = [self.parameters]
        self.resolution_choice = resolution_choice

    def model_name_definition(self) -> list[str]:
        """
        Return the model name for each parameters in self.list_parameters
        """

        return [parameters['physical model'] + '_' + parameters['optimization variable']
                for parameters in self.list_parameters]

    def parameters_simple_resolution(self) -> None:
        """
        Creates model name and file name
        """
        # Implementing the model name if necessary
        if 'model name' not in self.parameters:
            self.parameters['model name'] = self.model_name_definition()[0]
            # Implementing the file name if necessary
        if 'file name' not in self.parameters:
            if self.parameters['evolution save option']:
                self.parameters['file name'] = 'Evolution -' + self.parameters['model name']
            else:
                self.parameters['file name'] = self.parameters['model name']

    def parameters_simple_comparison(self, parameter_name: str, list_value: list[str | int | float | list | bool]
                                     ) -> None:
        """
        Changes the parameters in self.list_parameters. Also changes "model name" and "file name"
        :param parameter_name: string, the name of the varying parameter
        :param list_value: list of value taken by the varying parameter
        """

        # Creating the model name
        if 'model name' not in self.parameters:
            self.parameters['model name'] = self.model_name_definition()[0]

        # Creating the new list_parameters
        nb_model_to_compare = len(list_value)
        self.list_parameters = [self.parameters.copy()] * nb_model_to_compare

        # Initializing the file name
        list_model_name = []
        str_file_name = ''

        # Implementing the new parameters
        for ii in range(nb_model_to_compare):
            # Implementing the new dictionary and the value
            self.list_parameters[ii] = self.parameters.copy()

            value = list_value[ii]
            self.list_parameters[ii][parameter_name] = value

            # Changing the name of the model
            new_model_name = self.parameters["model name"] + str(value)
            self.list_parameters[ii]['model name'] = new_model_name

            str_file_name += str(value) + ','

            list_model_name.append(new_model_name)

        print("list model name", list_model_name)

        # Verification that there is no two exact model name:
        if len(set(list_model_name)) != len(list_model_name):
            print('You need to indicate a different model name for every model')
            return

        # Implementing the new file name if necessary
        if 'file name' not in self.list_parameters[0]:
            file_name = self.parameters["model name"] + ' - Var ' + str_file_name
            if len(file_name) > 100:
                file_name = file_name[:100]
            self.list_parameters[0]['file name'] = file_name
        print("list_parameters", self.list_parameters)

    def parameters_advanced_comparison(self, list_model_to_compare: list[dict[str, str | int | float | list | bool]]
                                       ) -> None:
        """
        Changes the parameters in self.list_parameters. Also changes "model name" and "file name"
        :param list_model_to_compare: list of dictionaries containing the parameters for each resolution
        """
        # Creating the new list_parameters
        nb_model_to_compare = len(list_model_to_compare)
        self.list_parameters = [self.parameters.copy()] * nb_model_to_compare
        list_model_name = []
        str_file_name = ''
        # Implementing the new parameters
        for ii in range(nb_model_to_compare):
            self.list_parameters[ii] = self.parameters.copy()
            model = list_model_to_compare[ii]

            # Implementing the new value
            for parameter_name in model:
                self.list_parameters[ii][parameter_name] = model[parameter_name]
            # Implementing the model name if necessary
            if 'model name' not in self.list_parameters[ii]:
                self.list_parameters[ii]['model name'] = self.model_name_definition()[ii]

            list_model_name.append(self.list_parameters[ii]['model name'])
            str_file_name += self.list_parameters[ii]['model name'] + ','

        # Verification that there is no two exact model name:
        if len(set(list_model_name)) != len(list_model_name):
            print('You need to indicate a different model name for every model')
            return

        # Implementation of the new file name if necessary
        if 'file name' not in self.list_parameters[0]:
            file_name = str_file_name
            self.list_parameters[0]['file name'] = file_name

    def verification_required_parameters(self) -> None:
        """
        Check that the dictionaries self.parameters and self.list_parameters are OK.
        If not, take default values for missing parameters.
        """
        list_required_parameters = ['index of profile',
                                    'albedo',
                                    'CO2',
                                    'number of levels',
                                    'physical model',
                                    'optimization variable',
                                    'plotting the graphics option']
        list_possible_parameters = ['file name',
                                    'model name',
                                    'feasibility',
                                    'feasibility objective function',
                                    "feasibility variable",
                                    'value to be equal to',
                                    'double step',
                                    'nb double step iteration',
                                    'linearisation of radiative flux',
                                    'function of the variable change',
                                    'maximal mass',
                                    'coefficient multiplication',
                                    'coefficient addition',
                                    'mass reference',
                                    'percent reference',
                                    'optimization variable to initialize',
                                    'initial value',
                                    'entropy variable',
                                    'positive entropy production',
                                    'graph',
                                    'optimization method',
                                    'maxiter of the minimize function',
                                    'resolution method',
                                    'max number of iterations',
                                    'variation threshold',
                                    # missing stuff for the different resolution methods
                                    'print option',
                                    "save option",
                                    'save to excel option',
                                    'initial value save option',
                                    'evolution save option',
                                    'nb iteration for saving and plotting',
                                    'plotting the graphics option',
                                    'list value to plot',
                                    'evolution plot option',
                                    'initial value plot option',
                                    "resolution choice"]
        list_all_parameters_coded = list_possible_parameters + list_required_parameters

        # Verification for all the model of the list_parameters
        for parameters in self.list_parameters:

            # Verification that the parameters entered are among the parameters that have been coded
            for parameter in parameters:
                if parameter not in list_all_parameters_coded:
                    print(
                        'The parameter ' + parameter + 'does not belong to the parameters that have been coded - check'
                                                       'for the exact spelling maybe?')

            # Verification of the required parameters that do not have default values
            for required_parameters in list_required_parameters:
                if required_parameters not in parameters:
                    print(
                        'the parameter ' + required_parameters + 'is missing in the parameters choice, it has to be '
                                                                 'added')

            # Verification and eventually initialisation of the required parameters
            if 'feasibility' not in parameters:
                parameters['feasibility'] = 'resolution'
            if 'linearisation of radiative flux' not in parameters:
                parameters['linearisation of radiative flux'] = 'once per resolution iteration'
            if 'optimization method' not in parameters:
                parameters['optimization method'] = 'SLSQP'
            if 'maxiter of the minimize function' not in parameters:
                parameters['maxiter of the minimize function'] = 500
            if 'resolution method' not in parameters:
                parameters['resolution method'] = 'threshold'
                parameters['max number of iterations'] = 30
                parameters['variation threshold'] = 10 ** (-9)
            if 'print option' not in parameters:
                parameters['print option'] = False
            if 'initial value save option' not in parameters:
                parameters['print option'] = False
            if 'evolution save option' not in parameters:
                parameters['evolution save option'] = False
            if 'list value to plot' not in parameters:
                parameters['list value to plot'] = ['E', 'T', 'F', 'q', 'M', 'P']
            if 'evolution plot option' not in parameters:
                parameters['evolution plot option'] = False
            if 'initial value plot option' not in parameters:
                parameters['initial value plot option'] = False

            # Verification of the optional parameters

            if 'test' in parameters['feasibility']:
                if 'feasibility objective function' not in parameters:
                    print(
                        "the parameter 'feasibility objective function' is missing in the parameters choice, "
                        "it has to be added")
                if 'feasibility objective function' in parameters:
                    if "feasibility variable" not in parameters:
                        parameters["feasibility variable"] = parameters["optimization variable"]
                        if parameters['feasibility objective function'] == parameters["sum equal"] \
                                and 'value to be equal to' not in parameters:
                            parameters["value to be equal to"] = 1

            if 'double step' in parameters:
                if 'nb double step iteration' not in parameters:
                    print(
                        "the parameter 'nb double step iteration' is missing in the parameters choice, it has to be "
                        "added")

            if 'm' in parameters['optimization variable']:
                if 'maximal mass' not in parameters:
                    print("the parameter 'maximal mass' is missing in the parameters choice, it has to be added")
                if 'function of the variable change' not in parameters:
                    print(
                        "the parameter 'function of the variable change' is missing in the parameters choice, it has to"
                        "be added")
                if 'tanh' in parameters['function of the variable change']:
                    list_function_variable_chang_parameters = ['coefficient multiplication', 'coefficient addition',
                                                               'mass reference', 'percent reference']
                    for function_variable_change_parameter in list_function_variable_chang_parameters:
                        if function_variable_change_parameter not in parameters:
                            print(
                                'the parameter ' + function_variable_change_parameter + 'is missing in the parameters '
                                                                                        'choice, it has to be added')

            if 'optimization variable to initialize' in parameters:
                if 'initial value' not in parameters:
                    print("the parameter 'initial value' is missing in the parameters choice, it has to be added")

            if parameters['evolution plot option']:
                if 'nb iteration for saving and plotting' not in parameters:
                    print(
                        "the parameter 'nb iteration for saving and plotting' is missing in the parameters choice, it has to be "
                        "added")

    def resolution(self) -> dict[str, dict[str, dict[str, np.ndarray | float | str]]]:
        """
        Resolve the problem depending on the choice of resolution, then plot and save.
        :return: results, a dict with 3 levels, results[model][instant][value_name]
        """
        type_resolution = self.resolution_choice['choice']
        # Particular case of the test
        if type_resolution == 'test':
            self.parameters_simple_resolution()
            self.verification_required_parameters()
            results = self.resolution_test()
            return results

        # Implementation of the list of parameters depending on the choice of comparison
        elif type_resolution == "simple resolution":
            self.parameters_simple_resolution()
        elif type_resolution == "simple comparison":
            parameter_name = self.resolution_choice['[if simple comparison] parameter that will variate ']
            list_value = self.resolution_choice['[if simple comparison] list of the values the parameter should take']
            self.parameters_simple_comparison(parameter_name, list_value)
        elif type_resolution == "advanced comparison":
            list_model_to_compare = self.resolution_choice['[if advanced comparison] list of the model ']
            self.parameters_advanced_comparison(list_model_to_compare)

        # Resolution
        self.verification_required_parameters()
        results = self.resolution_problem()
        # Saving and plotting the results
        results_save = self.shape_results("save", results, self.list_parameters)
        ps = PlotSave(results_save, self.list_parameters)
        if self.parameters['save to excel option']:
            ps.save_to_excel()
        if self.parameters["save option"]:
            ps.save()
        results_plot = self.shape_results("plot", results, self.list_parameters)
        ps = PlotSave(results_plot, self.list_parameters)
        if self.parameters['plotting the graphics option']:
            ps.plot()

        return results

    def resolution_problem(self) -> dict[str, dict[str, dict[str, np.ndarray | float | str]]]:
        """
        Call optimization.resolution_main() for each parameters' dictionary in self.list_parameters
        :return results: A dictionary of results for each parameters, corresponding to different model name
        """
        results = {}
        for parameters in self.list_parameters:
            optimization = Optimization(parameters)
            model_name = parameters['model name']

            # Particular case of the feasibility test
            if 'feasibility' in parameters and 'test' in parameters['feasibility']:
                result_feasibility = optimization.resolution_main()

                if parameters['feasibility'] == 'test and resolution':
                    parameters['feasibility'] = 'resolution'
                    if result_feasibility['final']['convergence']:
                        mep_phy = MepPhysics(parameters)
                        v_ini = mep_phy.create_initial_value(result_feasibility['final'], parameters["optimization variable"])
                        parameters['optimization variable to initialize'] = parameters['optimization variable']
                        parameters['initial value'] = v_ini

                        optimization = Optimization(parameters)
                        result_model = optimization.resolution_main()
                    else:
                        print(
                            'As the problem seems not to be feasible with the implementation choice, initial values '
                            'have not been implemented')
                        result_model = result_feasibility
                else:
                    result_model = result_feasibility

            # Particular case of the double step resolution
            elif 'double step' in parameters and parameters['double step']:
                nb_double_step_iteration = parameters['nb double step iteration']
                model_name_next_step = parameters['model name']

                for iterat in range(nb_double_step_iteration):
                    print('\n', '----- ITERATION ', iterat, ' FOR DOUBLE STEP -----')

                    # Step 1
                    results_intermediate = optimization.resolution_main()

                    # Initialisation Step 2
                    parameters = self.change_step_parameters(results_intermediate)

                    # Step 2
                    optimization = Optimization(parameters)
                    results_intermediate = optimization.resolution_main()
                    # parameters['model name'] = model_name_definition()

                    if iterat == nb_double_step_iteration - 1:
                        break
                    # Initialisation of the next Step 1 # you have to manually change it
                    mep_phy = MepPhysics(parameters)
                    xm_ini = mep_phy.create_initial_value(results_intermediate["final"], "xm")
                    parameters['optimization variable'] = 'xm'
                    parameters['optimization variable to initialize'] = 'xm'
                    parameters['initial value'] = xm_ini
                    parameters['model name'] = model_name_next_step
                    optimization = Optimization(parameters)

                # noinspection PyUnboundLocalVariable
                result_model = results_intermediate
                parameters['file name'] = 'resolution double step'

            # Normal resolution case
            else:
                result_model = optimization.resolution_main()
            results[model_name] = result_model

        return results

    # You have to change manually this function in order to get exactly what
    # you want in step 2 of 'double step'
    def change_step_parameters(self, results_intermediate):
        """
        Change the parameters for the next step. This function is designed to be manually modified.

        :param results_intermediate: the results of the last step
        :return: the parameters for the next step
        """
        mep_phy = MepPhysics(self.parameters)
        self.parameters['optimization variable to initialize'] = 'xmf'
        self.parameters['initial value'] = mep_phy.create_initial_value(results_intermediate["final"], "xmf")
        self.parameters['initial value'][1][0] = 10  # increase first mass flux
        self.parameters['maximal mass'] = np.inf

        return self.parameters

    @staticmethod
    def shape_results(plot_save_option, rs, list_parameters):
        """
        Keep in the results only what was asked plotting, or saving.

        :param plot_save_option: str, equal to "plot" or to "save"
        :param rs: results, the union of the results to be saved and plot
        :param list_parameters: the list of parameters for each resolution problem
        :return rs: the results, with unwanted parts removed
        """
        rs = deepcopy(rs)  # because of nested dicts
        if plot_save_option == "save":
            plot_save_list = ["plot", "save"]
        elif plot_save_option == "plot":
            plot_save_list = ["save", "plot"]
        else:
            print("Warning, you should use 'save' or 'plot' in shape_results function.")
        for ii in range(len(list_parameters)):
            # noinspection PyUnboundLocalVariable
            if list_parameters[ii][f"initial value {plot_save_list[0]} option"] and not list_parameters[ii][f"initial value {plot_save_list[1]} option"]:
                model_name = list(rs)[ii]
                del rs[model_name]["initial"]
            if list_parameters[ii][f"evolution {plot_save_list[0]} option"] and not list_parameters[ii][f"evolution {plot_save_list[1]} option"]:
                model_name = list(rs)[ii]
                to_remove = [xx for xx in rs[model_name].keys() if "int." in xx and xx.removeprefix("int.").isdigit()]
                for xx in to_remove:
                    del rs[model_name][xx]
        return rs

    def resolution_test(self) -> dict[str, dict[str, dict[str, np.ndarray | float | str]]]:
        results = {}

        for parameters in self.list_parameters:
            optimization = Optimization(parameters)
            model_name = 'test'
            results[model_name] = optimization.test(parameters)

        return results
