"""
MOTION CONTROLLER BASED ON INVERSE KINEMATICS
A basic end-effector pose controller for the robot arm.
It is only a kinematic control, producing end-effector velocities,
and tracking a (linear) trajectory in the task-space.

Example:
    Execute the kinematic motion control:
    $ uv run src/RobotArmIRB120/kinematicMotionControl.py

Author: Sebastian Heckers
Based on: Exercise 1c from 'Robot Dynamics', ETHZ
"""
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from RobotArmIRB120 import jointToPosition, jointToPosJac, pseudoInverseMat, viz_hist

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def kinematicMotionControl(q, r_des, v_des):
    """
    Compute the update joint velocities for velocity controllable robot.

    Input: q - current joint angle, r_des, v_des - desired end-effector position and velocity
    Output: Dq - joint-space velocity
    """
    K_p = 5  # linear position gain
    lam = 0.1  # pseudo-inverse damping coefficient

    r_ = jointToPosition(q)
    I_J = jointToPosJac(q)
    I_J_pinv = pseudoInverseMat(I_J, lam)

    dxe = r_des - r_
    w = v_des + K_p*dxe

    Dq = I_J_pinv @ w

    return Dq

def generateLineTrajectory(r_start, r_end, N):
    """
    Input: r_start, r_end - start & end position, N - number of timesteps
    Output: r_traj - list of end-effector position references (Nx3)
    """
    r_traj = np.linspace(r_start, r_end, N)
    return r_traj

if __name__ == "__main__":
    print("KINEMATRIC MOTION CONTROL\n A basic end-effector pose controller, producing end-effector velocities based on given trajectory.\n A linear trajectory across the domain is used.")
    # Trajectory settings
    t_s = 0.05  # sampling time [s]
    r_start = np.array([0.4, 0.1, 0.6])  # start point, 3x1 [m]
    r_end = np.array([-0.4, 0.3, 0.5])  # end point, 3x1 [m]
    v_line = 0.4  # velocity, 1x1 [m/s]
    q_0 = np.zeros(6)  # joint angle

    print(f"r_start = {r_start}\nr_end = {r_end}\nv_line = {v_line}\nt_s = {t_s}")

    q = q_0
    hist_ = []

    dr = r_end - r_start
    t_f = np.linalg.norm(dr)/v_line
    N = int(np.floor( t_f/t_s ))
    t = np.arange(1,N+1) * t_s

    r_traj = generateLineTrajectory(r_start, r_end, N)
    v_traj = np.ones((N,1)) @ (v_line * (dr/np.linalg.norm(dr))).reshape((1,3))

    for k in range(N):
        Dq = kinematicMotionControl(q, r_traj[k], v_traj[k])

        q = q + Dq*t_s

        hist_.append( (q, jointToPosition(q), jointToPosJac(q)@Dq) )

    ## PLOTTING
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['x', 'y', 'z']

    r_bot = np.array([r_ for q_,r_,v_ in hist_])
    v_bot = np.array([v_ for q_,r_,v_ in hist_])

    # Position
    fig, ax = plt.subplots()

    for i in range(3):
        ax.plot(t, r_bot[:, i],
                color=colors[i],
                label=labels[i])

        ax.plot(t, r_traj[:, i],
                '--',
                color=colors[i],
                label=f'{labels[i]}_ref')

    ax.set_title("End-effector position in inertial frame")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True)
    ax.legend()

    img_name = "/bin/kinematicMotionControl_pos.png"
    plt.savefig(repo_root+img_name)
    print(f"Saved plot: {img_name}")

    # Velocity
    fig, ax = plt.subplots()

    for i in range(3):
        ax.plot(t, v_bot[:, i],
                color=colors[i],
                label=labels[i])

        ax.plot(t, v_traj[:, i],
                '--',
                color=colors[i],
                label=f'{labels[i]}_ref')

    ax.set_title("End-effector linear velocity in inertial frame")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.grid(True)
    ax.legend()

    img_name = "/bin/kinematicMotionControl_vel.png"
    plt.savefig(repo_root+img_name)
    print(f"Saved plot: {img_name}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        r_bot[:, 0],
        r_bot[:, 1],
        r_bot[:, 2],
        label="Actual"
    )

    ax.plot(
        r_traj[:, 0],
        r_traj[:, 1],
        r_traj[:, 2],
        "--",
        label="Reference"
    )

    ax.set_title("End effector trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # Equal aspect ratio
    ax.set_box_aspect([
        np.ptp(r_bot[:, 0]),
        np.ptp(r_bot[:, 1]),
        np.ptp(r_bot[:, 2])
    ])

    ax.legend()
    ax.grid(True)

    img_name = "/bin/kinematicMotionControl_3D.png"
    plt.savefig(repo_root+img_name)
    print(f"Saved plot: {img_name}")

    ## Visualisation of Robot
    viz_hist([q_ for q_,dr_,dph_ in hist_], dt=t_s)