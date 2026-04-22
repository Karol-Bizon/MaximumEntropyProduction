# MEP

## Documentation:

The code is inline-commented and typed.

## Installation

This thread is written for Unix (more precisely Ubuntu 22) systems, but is easily adaptable to other systems.

#### Dependencies:

* git

#### Instructions:
* Make sure python3.11 and pip are installed `sudo apt-get -y install python3.11 python3.11-dev python3.11-venv python3-pip`
  * If `python3.11` cannot be found: `sudo apt install software-properties-common && sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update` and retry
* Install virtualenv: `python3.11 -m pip install --user virtualenv`
  * If you have issues with pip: `curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11` and retry
* If `(base)` is present in terminal, deactivate the conda environment : `conda deactivate` (type `conda activate` to undo)
* Create a virtualenv: `python3.11 -m virtualenv venv`
* Activate virtualenv: `source venv/bin/activate` (type `deactivate` to exit the virtualenv)
* Install dependencies: `python3.11 -m pip install -r requirements.txt`

You probably also want to compile the radiative code:
* `sudo apt-get install build-essential`
* `source compile_pybind11.sh` (create a file radiatif2.****.so, a module that can be imported in python.)

## Run the code

To run the code and tweak the parameters as you see fit, run main.py: `python main.py`

* To get the results from the main figure of the article, run main_figure.py: `python main_figure.py`
* To get the results from the figure with different CO2, run main_CO2_figure.py: `python main_CO2_figure.py`
 * To get the results from the figure comparing a local and a global maxima, run main_local_global_figure.py: `python main_local_global_figure.py`

#### Possible Errors:

* If you have a `No such file or directory` error, please check that there isn't any space in your absolute file names.
















