# -*- coding: utf-8 -*-
"""
CO2 sensitivity for a simplified tropical climate setup.

This script runs a series of historical and hypothetical CO2 scenarios
for one environmental profile (McClatchey tropical, index=1) and compares
vertical temperature distributions.
"""

from intermediate import Intermediate
from plot_save import PlotSave


parameters = {
    "index of profile": 1,  # tropical reference atmosphere
    "albedo": 0.1,
    "CO2": 280.0,
    "number of levels": 20,
    "physical model": "wc",
    "optimization variable": "xmf",  # keep h implicit (=1) for stable convergence
    "entropy variable": "x",
    "feasibility": "resolution",
    "feasibility objective function": "minimal sum",
    "feasibility variable": "x",
    "value to be equal to": 1,
    "double step": False,
    "nb double step iteration": 1,
    "linearisation of radiative flux": "once per resolution iteration",
    "function of the variable change": "simple multiplication",
    "maximal mass": 0.33,
    "coefficient multiplication": 0.1,
    "coefficient addition": 0,
    "mass reference": 0.003,
    "percent reference": 0.005,
    "optimization variable to initialize": "xmf",
    "initial value": [1, 0.2, 1],
    "positive entropy production": "Yes",
    "graph": "linegraph",
    "optimization method": "SLSQP",
    "maxiter of the minimize function": 2000,
    "resolution method": "threshold",
    "max number of iterations": 40,
    "variation threshold": 10 ** -9,
    "print option": True,
    "save option": True,
    "save to excel option": True,
    "plotting the graphics option": False,
    "plot reference data option": False,
    "initial value save option": False,
    "initial value plot option": False,
    "evolution save option": False,
    "evolution plot option": False,
    "nb iteration for saving and plotting": 1,
    # Keep a focused set of plots for period comparisons.
    "list value to plot": ["T", "F", "E", "P"],
    "resolution choice": "advanced comparison",
}

# Approximate representative values from paleoclimate literature + hypothetical tests.
# These are intentionally rough scenario markers, not strict reconstructions.
co2_scenarios_ppm = [
    ("LGM_21ka", 180.0),
    ("Preindustrial_1750", 280.0),
    ("Modern_2024", 420.0),
    ("High_2100_hyp", 800.0),
    ("PETM_56Ma", 1000.0),
    ("DeepTime_extreme_hyp", 2000.0),
]

list_model_to_compare = [
    {
        "CO2": co2_ppm,
        "model name": f"{label}_{int(co2_ppm)}ppm",
        "optimization variable to initialize": "xmf",
        "initial value": [1, 0.2, 1],
        "positive entropy production": "Yes",
    }
    for label, co2_ppm in co2_scenarios_ppm
]

resolution_choice = {
    "choice": parameters["resolution choice"],
    "[if simple comparison] parameter that will variate ": "CO2",
    "[if simple comparison] list of the values the parameter should take": [x[1] for x in co2_scenarios_ppm],
    "[if advanced comparison] list of the model ": list_model_to_compare,
}

inter = Intermediate(parameters, resolution_choice)
results = inter.resolution()

# Plot anomalies (delta) relative to modern reference, for converged models only.
reference_model_name = "Modern_2024_420ppm"
results_plot = inter.shape_results("plot", results, inter.list_parameters)
list_parameters_plot = []
results_plot_converged = {}
for model_parameters in inter.list_parameters:
    model_name = model_parameters["model name"]
    has_result = model_name in results_plot and "final" in results_plot[model_name]
    is_converged = has_result and bool(results_plot[model_name]["final"].get("convergence", False))
    if is_converged:
        model_parameters_plot = dict(model_parameters)
        model_parameters_plot["reference model name for differences"] = reference_model_name
        list_parameters_plot.append(model_parameters_plot)
        results_plot_converged[model_name] = results_plot[model_name]

if list_parameters_plot:
    if reference_model_name not in results_plot_converged:
        print(
            f"Warning: reference model '{reference_model_name}' did not converge. "
            "Using the first converged model as fallback reference."
        )
        list_parameters_plot[0]["reference model name for differences"] = list_parameters_plot[0]["model name"]
    ps_diff = PlotSave(results_plot_converged, list_parameters_plot, differences=True)
    ps_diff.plot()
else:
    print("No converged model to plot in anomaly mode.")

print("\n=== CO2 period scenarios summary ===")
for model_name, result in results.items():
    final = result["final"]
    t_surface = final["T"][0] if "T" in final else float("nan")
    entropy = final["entropy"] if "entropy" in final else float("nan")
    converged = final.get("convergence", False)
    print(
        f"{model_name:30s} | converged={str(converged):5s} | "
        f"sigma={entropy:.6e} W/m2/K | T_surface={t_surface:.2f} K"
    )
