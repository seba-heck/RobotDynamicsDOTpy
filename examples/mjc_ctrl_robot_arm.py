"""
MuJoCo Python BASIC EXAMPLE

created by Sebastin Heckers on 12.11.2023
based on the IsaacGym example: isaacgym/python/basic.py
"""

from datetime import datetime
import functools
from typing import Any, Dict, Sequence, Tuple, Union

import mujoco
import mujoco.viewer

import mediapy as media
import matplotlib.pyplot as plt

import random
import numpy as np
import os
import sys
import math
import argparse

from mjc_sim_lib import * # ProgramParameters
import RobotArmIRB120 as irb
import sympy as sp


# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

paused = False

def key_callback(keycode):
    print(keycode," ",chr(keycode))
    if chr(keycode) == ' ':
        print("Pause, ", paused)
        paused = not paused

def create_camera(gym,env,actor):
    cam_pos = gymapi.Vec3(-20.0,25.0,5.0)
    cam_target = gymapi.Vec3(0.0,5.0,1.0)
    cam_props = gymapi.CameraProperties()
    cam_props.width = 720
    cam_props.height = 720

    cam = gym.create_camera_sensor(env, cam_props)
    body = gym.get_actor_rigid_body_handle(env, actor, 0)
    gym.attach_camera_to_body(cam, env, body, gymapi.Transform(p=cam_pos), gymapi.FOLLOW_TRANSFORM)
    gym.set_camera_location(cam, env, cam_pos, cam_target)

    return cam

if __name__ == "__main__":
    print("# --- Prepare Simulation --- #")

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Input file for the scene.')
    # parser.add_argument('--config', type=str, default="config.yaml", help='Configuration file for the simulation.')
    parser.add_argument('--output', type=str, default="output.txt", help='Output file for the results.')
    parser.add_argument('--save_images', action="store_true", help='Store Images To Disk.')
    parser.add_argument('--headless', action="store_true", help='Run without viewer.')
    parser.add_argument('--solver_type', type=int, default=-1, help='Set solver type.')
    parser.add_argument('--integrator_type', type=int, default=-1, help='Set integrator type.')
    args = parser.parse_args()

    # check / make output path
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # make parameters
    params = ProgramParameters(args,__file__)

    q_des = np.ones(6)

    # LOADING KINEMATICS / EOM
    print(f"[START] Loading robot dynamics, EoM ...", end=" ", flush=True)
    bot = irb.IRB120()
    phi,dphi = irb.generalized_coords()
    kin = irb.generate_kinematics_sp(phi)
    jac = irb.generate_jacobian_sp(phi,dphi,kin,bot)

    J_ = jac["I_Je"]
    Jt = sp.lambdify(phi, J_, "numpy")
    I_dJe = sp.lambdify((*phi,*dphi), jac["I_dJe"], "numpy")
    I_Jpe = sp.lambdify(phi, jac["I_Jpe"], "numpy")
    eom = irb.load_EOM()

    H,E_kin,E_pot = irb.make_energies(eom)

    print(f"[DONE]")

    # load and compile model
    m = mujoco.MjModel.from_xml_path(args.scene)

    # set simulation parameters
    m.opt.integrator = params.integrator_type
    m.opt.solver = params.solver_type
    m.opt.timestep = 0.00001 # 0.000025

    # make data
    d = mujoco.MjData(m)

    # print model info
    print_sim_params(sys.stdout, params)
    print_model_info(sys.stdout, m)
    print_flex_info(sys.stdout, m)
    print_path_info(params.output_file, params)
    print_simulation_info(sys.stdout,m)

    # setup window
    if not params.flag_headless:
        print("# Make Viewer")
        viewer = mujoco.viewer.launch_passive(m, d, key_callback=key_callback, show_left_ui=False, show_right_ui=False)
        viewer.cam.lookat = [0,0,0]
        viewer.cam.distance = 20
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -5

    # Make renderer
    print("# Make Renderer")
    renderer = mujoco.Renderer(m)
    
    # variables
    step_counter = 0
    img_idx = 0
    prev_timestep = 0.0
    _next_timestep = params.dt_save
    q_i = d.qpos.copy()
    dq_i = d.qvel.copy()
    ddq_i = d.qacc.copy()
    t = d.time

    # make timers
    t_all = Timer()
    t_step = Timer()
    t_view = Timer()
    t_render = Timer()


    hist_ = [(q_i, dq_i, np.zeros(6), H(q_i,dq_i), E_kin(q_i,dq_i), E_pot(q_i), t)]

    print("# --- Start Simulation --- #")
    t_all.start_clock()

    # timeloop
    while (params.flag_headless or viewer.is_running()) and (params.flag_endless or t < params.total_time):

        g_i = np.asarray(eom["g"](*q_i), dtype=float).reshape(-1, 1)
        tau = irb.control_pd_g(q_des, q_i, dq_i, g_i).reshape(-1)
        d.ctrl = tau

        # step the physics
        t_step.start_clock()

        mujoco.mj_step(m, d)

        t_step.end_clock()

        # get current state
        q_i = d.qpos.copy()
        dq_i = d.qvel.copy()
        ddq_i = d.qacc.copy()
        t = d.time

        # print(f"(it {step_counter}/ t {t}) {q_i}")

        step_counter += 1

        # update the viewer
        if not params.flag_headless:
            t_view.start_clock()

            viewer.sync()

            t_view.end_clock()

        if (params.flag_draw_contacts):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(1)
        
        # render and save the pixels
        if (params.flag_save_images and (t - prev_timestep) > (1/params.fps)):
            t_render.start_clock()

            rgb_image_filename = params.image_path % (img_idx)
            if (img_idx % 10) == 0: print("Writing image %d (time %f): " % (img_idx,t),rgb_image_filename)
            
            renderer.update_scene(d)
            media.write_image(rgb_image_filename, renderer.render())
            
            img_idx += 1
            prev_timestep = t

            t_render.end_clock()

        # save current state
        if t >= _next_timestep:
            hist_.append((q_i, dq_i, ddq_i, H(q_i,dq_i), E_kin(q_i,dq_i), E_pot(q_i), t))
            _next_timestep += params.dt_save

        # stop program at maximum runtime
        if (not params.flag_endless and t_step.get_wall_time() > params.max_runtime):
            print("# STOPPED SIMULATION: reached maximal runtime")
            print(" Elapsed runime: %f sec." % (t_step.get_wall_time()))
            break

        # if step_counter >= 1000:
        #     break
    
    t_all.end_clock()

    if not params.flag_headless and viewer.is_running():
        viewer.close()

    renderer.close()

    print("# --- End Simulation --- #")

    print_result_info(params.output_file, params, t, step_counter, t_all, t_step, t_view, t_render)
    params.output_file.write(make_output_line())
    params.output_file.close()

    # SAVE HISTORY
    data = np.array([ np.concatenate([q, dq, ddq, [H, E_kin, E_pot, t]]) for q, dq, ddq, H, E_kin, E_pot, t in hist_ ])
    header = "q1 q2 q3 q4 q5 q6 dq1 dq2 dq3 dq4 dq5 dq6 ddq1 ddq2 ddq3 ddq4 ddq5 ddq6 H E_kin E_pot t"
    file_name = params.output_path + f"/hist_robotIRB120_control_test.txt"
    np.savetxt(file_name, data, header=header)
    print(f"Saved history: {file_name}")

    # print elapsed time
    print("# Measured Times")
    print(" Wall-Time: %f sec." % (t_all.get_wall_time()))
    print(" CPU-Time: %f sec." % (t_all.get_cpu_time()))
    print(" Simulation-Step Wall-Time: %f sec." % (t_step.get_wall_time()))
    print(" Simulation-Step CPU-Time: %f sec." % (t_step.get_cpu_time()))

    print("# Simulation Time")
    print(" Simulation-Time: %f" % (t))
    print(" Real-Time-factor: %f" % (t/t_all.get_cpu_time()))

    # glfw.terminate()

    print("# --- Finish Program --- #")
    sys.exit()