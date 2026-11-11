import os
import numpy as np
import matplotlib.pyplot as plt

from RobotArmIRB120 import jointToPosition


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


colors = ['tab:blue', 'tab:orange', 'tab:green']
labels = ['H', 'E_kin', 'E_pot']


# Energy
def plotting_history_energy(t,hist_,img_name="/bin/hist_energy.png"):
    fig, ax = plt.subplots()

    ax.plot(t, [h for _,_,_,h,_,_,_ in hist_], color='tab:red', label='H')
    ax.plot(t, [h for _,_,_,_,h,_,_ in hist_], color='tab:blue', label='E_kin')
    ax.plot(t, [h for _,_,_,_,_,h,_ in hist_], color='tab:green', label='E_pot')

    ax.set_title("Energy Diagram")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy")
    ax.grid(True)
    ax.legend()

    plt.savefig(repo_root+"/"+img_name)
    print(f"Saved plot: {img_name}")

# Joint Angle
def plotting_history_joints(t,hist_,img_name="/bin/hist_q.png"):
    fig, ax = plt.subplots()

    ax.plot(t, [list(h) for h,_,_,_,_,_,_ in hist_], label=[f"q_{i+1}" for i in range(6)])

    ax.set_title("Joint Angle Diagram")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [rad]")
    ax.grid(True)
    ax.legend()

    plt.savefig(repo_root+"/"+img_name)
    print(f"Saved plot: {img_name}")

# Position
def plotting_history_pos(t,hist_,img_name="/bin/hist_pos.png",x_des=None):
    fig, ax = plt.subplots()

    x_bot = np.array([list(jointToPosition(h)) for h,_,_,_,_,_,_ in hist_])
    # if x_des is not None:
    #     if controller == "PDg":
    #         x_des = 
    #     else: # controller == "PD"
    #         x_des = np.array([list(_move_r_des(t_i)) for t_i in t])
    #         # x_des = np.array([list(r_des) for h,_,_,_,_,_,_ in hist_])
    for i in range(3):
        ax.plot(t, x_bot[:,i], label=f"x_{i+1}", color=colors[i])
        if x_des is not None:
            ax.plot(t, x_des[:,i], '--', label=f"x_des_{i+1}", color=colors[i])

    ax.set_title("End-effector Position Diagram")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True)
    ax.legend()

    plt.savefig(repo_root+"/"+img_name)
    print(f"Saved plot: {img_name}")
