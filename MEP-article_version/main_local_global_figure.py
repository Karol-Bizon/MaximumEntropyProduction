# -*- coding: utf-8 -*-
"""
Main code. Change parameters
"""

# noinspection PyUnusedLocal
import numpy as np
from intermediate import Intermediate

parameters = {'index of profile': 1,
              'albedo': 0.1,
              'CO2': 280.0,
              'number of levels': 20,
              'physical model':
              "wc",

              'linearisation of radiative flux':
                  "once per resolution iteration",

              # [if 'm' in optimization variable]
              'function of the variable change':
                  'simple multiplication',
              'optimization variable to initialize': "xmf",

              'graph':
              "linegraph",

              'optimization method': 'SLSQP',  # 'SLSQP', #'trust-constr', #'matrix'
              'maxiter of the minimize function': 2000,

              'resolution method': 'threshold',
              'max number of iterations': 40,
              'variation threshold': 10 ** -9,

              'print option': True,
              "save option": True,
              'save to excel option': True,
              'plotting the graphics option': True,
              'initial value save option': False,
              'list value to plot': ['F_dry', 'F_wet', 'E_dry', 'E_wet', 'E', 'T', 'F', 'q', 'M', 'P', 'theta', 'h'],

              "resolution choice":
                     "advanced comparison",
              }

# Advanced Comparison
list_model_to_compare = [
    {'physical model': 'wc', 'optimization variable': 'xmf', 'initial value': [1, 0.2, 1],
                 'positive entropy production': 'Yes', 'maximal mass': 0.33, 'coefficient multiplication': 0.1,
                 'model name': "global max"
     },
    {'physical model': 'wc', 'optimization variable': 'xmf', 'initial value': [1, 0.2, 0.1],
                 'positive entropy production': 'Yes', 'maximal mass': 0.33, 'coefficient multiplication': 0.1,
                 'model name': "local max"
     }
                         ]

# Simple comparison
parameter_name = 'CO2'
list_value = [180, 280, 560]

resolution_choice = {'choice':
                     parameters["resolution choice"] if "resolution choice" in parameters else "simple resolution",
                     '[if simple comparison] parameter that will variate ': parameter_name,
                     '[if simple comparison] list of the values the parameter should take': list_value,
                     '[if advanced comparison] list of the model ': list_model_to_compare}

inter = Intermediate(parameters, resolution_choice)
results = inter.resolution()

# Documentation of the parameters #

# MODELING PARAMETERS
# index of profile                  int          between 1 and 5, see profile bis
# albedo                            float        0.1 for profile 1, 0.6 for the others?
# CO2                               float
# number of levels                  int          more than 2
# physical model                    str          'un', 'ini_moist', 'ini_dry', 'ini_CpT', 'wc'
#
# OPTIMIZATION AND RESOLUTION PARAMETERS
#
# optimization variable             str          concatenation of 'x', 'm', 'c', 'f' or 'h'
# entropy variable                  str          [only for ini_xmf or ini_xm] 'x', 'xf', 'xm'
#
# feasibility                       str          'test', 'test and resolution' or 'resolution' - optional
# feasibility objective function    str          [for 'test' and 'test and resolution'] 'null', 'minimal sum', 'maximal sum', 'sum equal'
# feasibility variable              str          concatenation of 'x', 'm', 'c', 'f' or 'h'
# value to be equal to              float/array
#
# linearisation of radiative flux   bool     'once per resolution iteration', 'no linearisation'
#
# function of the variable change   str          [if 'm' in optimization variable] 'no', 'simple multiplication', 'tanh', '1-exp'
# maximal mass                      fl
# coefficient multiplication        fl
# coefficient addition              fl           [if 'tanh'/'1-exp'] -
# mass reference                    fl           [if 'tanh'/'1-exp'] -
# percent reference                 fl           [if 'tanh'/'1-exp']
#
# optimization variable to initialize  str       {OPTIONAL} concatenation of 'x', 'm', 'c', 'f' or 'h'
# initial value                     fl/array     {OPTIONAL} (the value that will be set, if it is a
# float, all the layers will have the same value, if it is a list of np.arrays, they have to describe all the layers)
#
# positive entropy production       str          'Yes, 'No', (add the constraint that the global entropy production is
# > 0. It can help converging sometimes to bound the entropy)
#
# graph                             str          'linegraph', 'doublestargraph', (The doublestargraph is a stargraph
# linked once to the vertex number 0.)
#
# optimization  method              str          'SLSQP', 'trust-constr', 'matrix' (matrix is for unconstrained only)
# maxiter of the minimize function  int          (the maximum number of iteration allowed to the
# scipy.optimize.minimize algorithm)
#
# resolution method                 str          'simple', 'convergence', 'threshold', 'tabu', 'maxiter variation', 'value variation'
# number of iterations              int          [for 'maxiter variation', 'value variation']
# max number of iterations          int          [for 'simple', 'convergence', 'threshold', 'tabu']
# variation threshold               float        [for 'threshold']
# list maxiter                      list fl/fl   [for 'maxiter variation'] (the min and max maxiter)
# variable to change                str          [for 'value variation'] 'x', 'm', 'f'
# min-max of the variable           (fl,fl)      [for 'value variation'] (the min and max
# power of 10 tolerated for the value)
# number of iteration per cycle     int          [for 'value variation'] (the number of interation per cycle)
#
#
# PRINT SAVE PARAMETERS
# print option                      bool         if True, will print all the information while algorithm running
# save option                       bool         if True, will save parameters and results in json and npy files
# save to excel option              bool         if True, will save results in an Excel file
# plotting the graphics option      bool         if True, plot the results
# initial value save option         bool         if True, save the initial value
# initial value plot option         bool         if True, plot the initial value
# evolution save option             bool         if True, will save all the k iterations of the resolution algorithm
# evolution plot option             bool         if True, will plot all the k iteration
# nb iteration for saving and plotting  int      [if evolution save option == True] : the k parameters
# list value to plot                list         (possible values:
# 'v', 'F_dry', 'F_wet', 'E_dry', 'E_wet', 'T', 'M', 'z', 'A', 'F', 'P', 'theta', 'q', 'h' and constraints)
