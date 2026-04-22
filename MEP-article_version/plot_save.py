# -*- coding: utf-8 -*-
"""
Plot and Save.
"""

from mep_physics import MepPhysics
import profile_bis as prf
import constants as cst

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import itertools
import sys
import shutil
import json
import os

value_parameter_names = {"v": {"value name": "v"},
                         'T': {'value name': 'Temperature', 'unit': 'T (K)'},
                         'theta': {'value name': 'Potential Temperature', 'unit': 'no unit'},
                         'E': {'value name': 'Specific Energy ()', 'unit': ' e (kJ.kg^{-1})'},
                         'E_dry': {'value name': 'Specific Energy (dry)', 'unit': 'e (kJ.kg^{-1})'},
                         'E_wet': {'value name': 'Specific Energy (wet)', 'unit': 'e (kJ.kg^{-1})'},
                         'q': {'value name': 'Absolute humidity', 'unit': 'no unit'},
                         'h': {'value name': 'Relative Humidity', 'unit': 'no unit'},
                         'F': {'value name': 'Energy Flux ()', 'unit': 'F (W.m^{-2})'},
                         'F_dry': {'value name': 'Energy Flux (dry)', 'unit': 'F (W.m^{-2})'},
                         'F_wet': {'value name': 'Energy Flux (wet)', 'unit': 'F (W.m^{-2})'},
                         'M': {'value name': 'Mass Flux', 'unit': 'm (kg.m^{-2}.s^{-1})'},
                         'A': {'value name': 'Mass Flux (advective)', 'unit': 'a (kg.m^{-2}.s^{-1})'},
                         'P': {'value name': 'Precipitation', 'unit': 'P (m/an)'},
                         'entropy': {'value name': 'Entropy', 'unit': 'test'},
                         'feasibility objective function': {"value name": "Feasibility Objective Function"},
                         'radiative flux': {'value name': 'Radiative Flux', 'unit': 'W.m^{-2}'},
                         'x': {'value name': 'x (inverse of temperature)', 'unit': 'no unit'},
                         'z': {'value name': 'z (function of the mass)', 'unit': 'no unit'},
                         'con_local_energy_balance': {'value name': 'Cons. local energy balance'},
                         'con_local_energy_balance_error': {'value name': 'Cons. local energy balance Error'},
                         'con_global_energy_balance': {'value name': 'Cons. global energy balance'},
                         'con_global_energy_balance_error': {'value name': 'Cons. global energy balance Error'},
                         'con_pos_m': {'value name': 'Cons. positivity of the mass'},
                         'con_pos_p': {'value name': 'Cons. positivity of the precipitation'},
                         'con_pos_p_error': {'value name': 'Cons. positivity of the precipitation Error'},
                         'con_pos_alpha': {'value name': 'Cons. positivity of alpha'},
                         'con_pos_alpha_error': {'value name': 'Cons. positivity of alpha Error'},
                         'con_mass_balance_on_boundary_layer': {
                             'value name': 'Cons. mass balance on the boundary layer'},
                         # 'con_sign': {'value name': 'constraint on the sign for dst'},
                         'con_R_null': {'value name': 'constraint on R null'},
                         'con_def_convective_flux': {'value name': 'Cons f definition'},
                         'con_def_convective_flux_error': {'value name': 'Cons f definition Error'},
                         'con_possibility_precipitation': {'value name': 'Cons possibility of precipitation'},
                         'convergence': {'value name': 'convergence'},
                         "number of iteration": {"value name": "number of iteration"},
                         'mass balance': {'value name': 'mass balance'}}


class PlotSave:
    """
    The plotting process is in this order:
    _ taking a value (T, F, M, E...)
    _ taking parameters in list_parameters
    _ taking an instant
    _ plot
    """

    def __init__(self, results: dict, list_parameters: list[dict], differences: bool = False):
        self.results = results
        self.list_parameters = list_parameters
        self.differences = differences

        self.x_min_max = True
        self.forma = "jpg"
        self.fontsize = 12
        self.dpi = 600
        self.show_plot = False

    def plot(self) -> None:
        """
        Plotting with the pressure as coordinate, for every physical value.

        :return: None
        """
        value_choice = self.list_parameters[0]['list value to plot']

        for value_name in value_choice:
            if "stargraph" in self.list_parameters[0]["graph"] and value_name in ["M", "A", "F", "F_dry", "F_wet"]:
                for plot_scatter in [False, True]:
                    self.plot_pressure(value_name, plot_scatter)
            else:
                self.plot_pressure(value_name)

        plt.show()

    def plot_pressure(self, value_name: str, plot_scatter: bool = False) -> None:
        """
        Plotting with the pressure as coordinate, for a given physical value.

        :param value_name: T, E, M, A, F, q, h ...
        :param plot_scatter: True or False. Plot a scatter plot for a star or doublestargraph.
        :return: None
        """
        print("value", value_name)
        plt.style.use('tableau-colorblind10')
        fig, ax = plt.subplots()

        str_model_name = ''
        # Using different lines style and markers when plotting several curves in a fig
        marker = itertools.cycle(("s", "D", ">", "o", "<"))  # ,'+','+','+','o','o','o','o','o',\
        # line = itertools.cycle(["-", "--", "-.", ":"])
        line = itertools.cycle([""])
        markerSize = itertools.cycle([10 * size for size in [1, 0.9, 1.1, 1.05, 1.1]])

        # used to plot differences instead of absolute
        if self.differences and len(self.list_parameters) >= 2:  # if plotting differences
            absoluteValue = self.results[self.list_parameters[1]['model name']]['final'][value_name]

        for parameters in self.list_parameters:
            mep_phy = MepPhysics(parameters)
            n = parameters["number of levels"]
            pressure = prf.pressureScale(cst.p0, n)  # pressure levels, middle of boxes = (ps, p1, p2 ...p_nLevels)
            pressureB = prf.pressureBounds(pressure)

            if parameters['graph'] == 'doublestargraph':
                start, target, _ = mep_phy.doublestargraph(mep_phy.N)
                # return the indices in 'start' of the edges corresponding to a linegraph
                indexStartLine = [idx for idx, item in enumerate(start) if item not in start[:idx]]
            elif parameters['graph'] == 'stargraph':
                start, target, _ = mep_phy.stargraph(mep_phy.N)
                # return the indices in 'start' of the edges corresponding to a linegraph
                indexStartLine = [idx for idx, item in enumerate(start) if item not in start[:idx]]
            elif parameters['graph'] == 'linegraph':
                start, target, _ = mep_phy.linegraph()

            # Declaration of the model and the option
            model = parameters['model name']
            str_model_name += ',' + model
            evolution_plot_option = parameters['evolution plot option']
            initial_plot_option = parameters['initial value plot option']

            # a function to plot fluxes depending on whether "graph" is "line" or "star"
            def plot_flux(value_plot_f: np.ndarray, ax_f: object) -> None:
                """
                Plotting flux like F, m, or a.

                :param value_plot_f: 1D ndarray of dim N
                :param ax_f: axes from subplots()
                :return: None
                """
                if parameters["graph"] == "linegraph":
                    ax_f.plot(value_plot_f, pressureB[:n], linestyle=next(line), label=label, marker=next(marker),
                              linewidth=1, markersize=next(markerSize))
                elif parameters['graph'] in ['stargraph', 'doublestargraph']:
                    x_values_flux_line = value_plot_f[indexStartLine]
                    x_values_flux_start = np.zeros(n)
                    x_values_flux_target = np.zeros(n)
                    x_values_flux_total = np.zeros(n)
                    for ii in range(len(start)):
                        x_values_flux_start[int(start[ii])] += value_plot_f[ii]  # start
                        x_values_flux_target[int(target[ii]) - 1] += value_plot_f[ii]  # target
                        for jj in range(int(start[ii]), int(target[ii])):
                            x_values_flux_total[jj] += value_plot_f[ii]  # total
                    ax_f.plot(x_values_flux_line, pressureB[:n], linestyle=next(line), label=label,
                              marker=next(marker), linewidth=1)
                    ax_f.plot(x_values_flux_start, pressureB[:n], linestyle=next(line), label='start',
                              marker=next(marker), linewidth=1)
                    ax_f.plot(x_values_flux_target, pressureB[:n], linestyle=next(line), label='target',
                              marker=next(marker), linewidth=1)
                    ax_f.plot(x_values_flux_total, pressureB[:n], linestyle=next(line), label='total',
                              marker=next(marker), linewidth=1)

            # Creating the list of the instant to plot
            list_temp = []
            if evolution_plot_option:
                list_temp += list(self.results[model].keys())
            elif initial_plot_option:
                list_temp += ['initial']
                list_temp += ['final']
            else:
                list_temp += ['final']

            if value_name == 'P':
                pressure = pressure[1:]

            # Plotting for every instant
            for instant in list_temp:
                str_entropy = r' $\sigma =$' + "%.5e" % self.results[model][instant]['entropy'] + ' $W/m^2/K$'
                if evolution_plot_option or initial_plot_option:
                    label = model + instant + str_entropy
                else:
                    label = model + ',' + str_entropy  # to uncomment

                # Verifying the value can be plotted and then plotting it
                if value_name not in self.results[model][instant]:
                    print('The value', value_name,
                          ' can not be plot because there is no result associated with this value')
                else:
                    value_plot = self.results[model][instant][value_name]
                    # plotting differences
                    if self.differences and len(self.list_parameters) >= 2:  # plot the differences
                        # noinspection PyUnboundLocalVariable
                        value_plot = value_plot - absoluteValue
                    if value_name in ['E', 'E_dry', 'E_wet']:
                        value_plot = value_plot / 1000  # J/kg -> kJ/kg
                    if value_name in ['M', 'z', 'A', "F", "F_dry", "F_wet"]:
                        # Plot for a doublestargraph
                        if plot_scatter:
                            # noinspection PyUnboundLocalVariable
                            ax.scatter(start, target, c=value_plot, vmax=0.33, cmap='bwr')
                        else:
                            plot_flux(value_plot, ax)
                    elif value_name == "v":
                        plt.close(fig)
                        return
                    else:
                        ax.plot(value_plot, pressure, linestyle=next(line), label=label,
                                marker=next(marker), linewidth=1, markersize=next(markerSize))

        # plot extra data
        if value_name == 'T' and not self.differences:  # plotting absolute values
            ax.plot(prf.McClatcheyProfileTropical.temperature, prf.McClatcheyProfileTropical.pressure,
                    marker=next(marker), label='obs (McClatchey 1972)', linestyle="",
                    markersize=next(markerSize))
            ax.plot(np.loadtxt("data_IPSL_CM6A_LR/IPSL_CM6A_LR_temp.txt"),
                    np.loadtxt("data_IPSL_CM6A_LR/IPSL_CM6A_LR_plev.txt") / 100, marker=next(marker),
                    label='IPSLCM6A_LR', linestyle="", markersize=next(markerSize))
        if value_name == 'T' and self.differences:  # plotting differences
            ax.plot(-np.loadtxt("data_IPSL_CM6A_LR/IPSL_CM6A_LR_temp.txt") +
                    np.loadtxt("data_IPSL_CM6A_LR/IPSL_CM6A_LR_temp_ssp245.txt"),
                    np.loadtxt("data_IPSL_CM6A_LR/IPSL_CM6A_LR_plev.txt") / 100, marker=next(marker),
                    label='IPSLCM6A_LR', linestyle="", markersize=next(markerSize))
        if value_name == "h" and not self.differences:
            ax.plot(np.loadtxt("data_IPSL_CM6A_LR/IPSL_CM6A_LR_hur.txt") / 100,
                    np.loadtxt("data_IPSL_CM6A_LR/IPSL_CM6A_LR_plev.txt") / 100, marker=next(marker),
                    label="IPSLCM6A_LR", linestyle="", markersize=next(markerSize))

        # Plotting the legend
        plt.rcParams.update({'font.family': 'serif'})
        if plot_scatter:
            colorbar_label = value_parameter_names[value_name]['value name'] + ' $' + value_parameter_names[value_name][
                'unit'] + '$ '
            fig.colorbar(mappable=ax.collections[0]).set_label(colorbar_label, fontsize=self.fontsize)
            # fig.colorbar().set_label(colorbar_label, fontsize=self.fontsize)
            xlabel = "starting box"
            ylabel = "ending box"
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            xlabel = value_parameter_names[value_name]['value name'] + ' $' + value_parameter_names[value_name][
                'unit'] + '$ '
            ylabel = 'Pressure $P(mB)$'
            ax.set_ylim([cst.p0, 0])
        ax.set_xlabel(xlabel, fontsize=self.fontsize)
        ax.set_ylabel(ylabel, fontsize=self.fontsize)
        ax.tick_params(labelsize=self.fontsize)

        if self.x_min_max:
            if value_name == 'T':
                # ax.set_xlim([-5,5])
                ax.set_xlim([180, 320])
            if value_name == 'F':
                ax.set_xlim([-15, 120])
            if value_name == 'E':
                ax.set_xlim([300, 520])
            if value_name == 'M':
                # ax.set_xlim([-0.05,min(10,parameters['maximal mass'])])
                ax.set_xlim([-0.05, 0.33])
                # ax.set_xlim([-0.01,0.2])
            if value_name == 'P':
                # ax.set_xlim([-0.5,100])
                # ax.set_xlim([-0.5,18])
                ax.set_xlim([-0.1, 3])

        figure_title = value_parameter_names[value_name]['value name']
        ax.set_title(figure_title)
        if not plot_scatter:
            ax.legend(loc="lower left", borderaxespad=0.9, fontsize=self.fontsize)  # to uncomment

        # Saving
        folder = 'results'
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_title = figure_title + ' - ' + self.list_parameters[0]['file name']
        if plot_scatter:
            filename = str(folder + '/./' + file_title + '_scatter' + '.' + self.forma)
        else:
            filename = str(folder + '/./' + file_title + '.' + self.forma)
        fig.tight_layout()
        fig.savefig(filename, dpi=self.dpi, format=self.forma)
        plt.show(block=False)
        if not self.show_plot:
            plt.close(fig)

    def save_to_excel(self) -> None:
        dict_pd = {}

        for i in range(len(self.list_parameters)):
            model = self.list_parameters[i]['model name']
            evolution_save_option = self.list_parameters[i]['evolution save option']
            initial_save_option = self.list_parameters[i]['initial value save option']

            for temp in self.results[model]:
                for value in list(self.results[model][temp]):
                    column_figures = self.results[model][temp][value]
                    if type(column_figures) is not np.ndarray:
                        column_figures = [column_figures]

                    if evolution_save_option or initial_save_option:
                        column_name = value_parameter_names[value]['value name'] + '-' + model + '-' + temp
                    else:
                        column_name = value_parameter_names[value]['value name'] + '-' + model

                    dict_pd[column_name] = column_figures

        folder = 'results'
        title = 'Results -  ' + self.list_parameters[0]['file name'][:150]
        complete_file_name = str(folder + '/./' + title + '.xlsx')
        # complete_file_name2 = str(folder + '/./' + title+'.odf')
        df = pd.DataFrame.from_dict(dict_pd, orient="index")
        df = df.transpose()
        df.to_excel(complete_file_name)
        # df.to_excel(complete_file_name2)

    def save(self) -> None:
        class NumpyEncoder(json.JSONEncoder):
            """
            For storing numpy arrays with json.
            """

            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        savePath = "results"

        # save results
        np.save(os.path.join(savePath, "results.npy"), self.results)

        # save parameters
        file = open(os.path.join(savePath, "list_parameters.json"), "w")
        print(self.list_parameters)
        json.dump(self.list_parameters, file, indent=1, cls=NumpyEncoder)
        file.close()

        # save current simulation file to keep a trace of what we ran
        orig_file = sys.modules['__main__'].__file__
        shutil.copyfile(orig_file, os.path.join(savePath, "source.py"))


if __name__ == "__main__":
    save_path = "results"
    differences = False

    results = np.load(os.path.join(save_path, "results.npy"), allow_pickle=True).item()
    f = open(os.path.join(save_path, "list_parameters.json"), "r")
    list_parameters = json.load(f)

    # noinspection PyTypeChecker
    ps = PlotSave(results, list_parameters, differences=differences)
    ps.plot()
