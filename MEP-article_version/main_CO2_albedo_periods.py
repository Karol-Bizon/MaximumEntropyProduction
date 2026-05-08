# -*- coding: utf-8 -*-
"""
Combined CO2 + albedo period sensitivity.

Robust mode:
- sequential scenario solving,
- warm start from previous converged case,
- automatic retry with trust-constr when SLSQP fails.
"""

from copy import deepcopy

from intermediate import Intermediate
from mep_physics import MepPhysics
from plot_save import PlotSave


parameters = {
    "index of profile": 6,  # mid-latitude mean reference atmosphere (better global proxy)
    "albedo": 0.3,
    "CO2": 420.0,
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
    "file name": "alb_co2",
    "compact file naming option": True,
    "results subfolder": "period_runs/alb_co2",
    "plotting the graphics option": False,
    "plot reference data option": False,
    "initial value save option": False,
    "initial value plot option": False,
    "evolution save option": False,
    "evolution plot option": False,
    "nb iteration for saving and plotting": 1,
    "list value to plot": ["T", "F", "E", "P"],
}

# Paired scenarios by period:
# - CO2 varies with period
# - albedo also varies with period (higher in glacial, lower in warm low-ice worlds)
period_co2_albedo = [
    ("LGM_21ka", 180.0, 0.34),
    ("Preindustrial_1750", 280.0, 0.30),
    ("Modern_2024", 420.0, 0.30),
    ("High_2100_hyp", 800.0, 0.29),
    ("PETM_56Ma", 1000.0, 0.27),
    ("DeepTime_extreme_hyp", 2000.0, 0.26),
]

list_model_to_compare = [
    {"CO2": co2_ppm, "albedo": albedo, "model name": f"{label}_{int(co2_ppm)}ppm_alb{albedo:.2f}"}
    for label, co2_ppm, albedo in period_co2_albedo
]


def run_single_model(base_parameters: dict, model_overrides: dict, warm_start: list | None):
    base = deepcopy(base_parameters)
    base.update(model_overrides)
    base["save option"] = False
    base["save to excel option"] = False
    base["plotting the graphics option"] = False
    base["optimization variable to initialize"] = base["optimization variable"]
    if warm_start is not None:
        base["initial value"] = warm_start

    attempts = [
        ("SLSQP", {"optimization method": "SLSQP"}),
        (
            "trust-constr",
            {
                "optimization method": "trust-constr",
                "maxiter of the minimize function": 4000,
                "max number of iterations": 80,
            },
        ),
    ]

    last_params = None
    last_result = None
    used_method = "none"
    for method_name, method_patch in attempts:
        params_try = deepcopy(base)
        params_try.update(method_patch)
        inter_try = Intermediate(params_try, {"choice": "simple resolution"})
        result_try = inter_try.resolution()
        model_name = params_try["model name"]
        final_try = result_try[model_name]["final"]
        last_params = params_try
        last_result = result_try[model_name]
        used_method = method_name
        if bool(final_try.get("convergence", False)):
            break

    return last_params, last_result, used_method


results = {}
used_methods = {}
final_parameters_list = []
warm_start_value = None

for model in list_model_to_compare:
    params_used, result_model, method_name = run_single_model(parameters, model, warm_start_value)
    model_name = params_used["model name"]
    results[model_name] = result_model
    used_methods[model_name] = method_name
    final_parameters_list.append(params_used)

    if bool(result_model["final"].get("convergence", False)):
        mep_phy = MepPhysics(params_used)
        warm_start_value = mep_phy.create_initial_value(result_model["final"], params_used["optimization variable"])

if parameters["save option"]:
    PlotSave(results, final_parameters_list).save()
if parameters["save to excel option"]:
    PlotSave(results, final_parameters_list).save_to_excel()

# Plot anomalies (delta) relative to modern reference, for converged models only.
reference_model_name = "Modern_2024_420ppm_alb0.30"
list_parameters_plot = []
results_plot_converged = {}
for model_parameters in final_parameters_list:
    model_name = model_parameters["model name"]
    has_result = model_name in results and "final" in results[model_name]
    is_converged = has_result and bool(results[model_name]["final"].get("convergence", False))
    if is_converged:
        model_parameters_plot = dict(model_parameters)
        model_parameters_plot["reference model name for differences"] = reference_model_name
        list_parameters_plot.append(model_parameters_plot)
        results_plot_converged[model_name] = results[model_name]

if list_parameters_plot:
    if reference_model_name not in results_plot_converged:
        print(
            f"Warning: reference model '{reference_model_name}' did not converge. "
            "Using the first converged model as fallback reference."
        )
        list_parameters_plot[0]["reference model name for differences"] = list_parameters_plot[0]["model name"]
    PlotSave(results_plot_converged, list_parameters_plot, differences=True).plot()
else:
    print("No converged model to plot in anomaly mode.")

print("\n=== Combined CO2 + albedo period scenarios summary (robust mode) ===")
for model_name, result in results.items():
    final = result["final"]
    t_surface = final["T"][0] if "T" in final else float("nan")
    entropy = final["entropy"] if "entropy" in final else float("nan")
    converged = final.get("convergence", False)
    method = used_methods.get(model_name, "unknown")
    print(
        f"{model_name:30s} | converged={str(converged):5s} | method={method:11s} | "
        f"sigma={entropy:.6e} W/m2/K | T_surface={t_surface:.2f} K"
    )
