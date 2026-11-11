from .timer import Timer
from .lib_mujoco import *
from .plotting import *

__all__ = [
    'Timer',
    # mujoco helper
    'integratorNames',
    'solverNames',
    'ProgramParameters',
    'ControlPDgParameters',
    'ControlINVParameters',
    'ControlOPSParameters',
    'CTRL_CLASS',
    'print_path_info',
    'print_result_info',
    'print_sim_params',
    'print_model_info',
    'print_flex_info',
    'print_simulation_info',
    'make_output_line',
    # plotting functions
    'plotting_history_energy',
    'plotting_history_joints',
    'plotting_history_pos',
]

__version__ = "0.1.0"
__author__ = "Sebastian Heckers"

print(f"Common module of RobotDynamics.py, version {__version__} successfully loaded.")