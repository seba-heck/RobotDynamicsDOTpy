"""
SCRIPT for MODEL-BASED CONTROL of ROBOT ARM IRB120
This script runs different model-based controllers (or none) and simulate
the motion of it. It uses different integration methods, plots the results,
and visualizes the robot motion.

Example:
    Running the script:
    $ uv run examples/evaluate_control_arm.py
    ...

Notes:
    Set the configuration parameters for each run:
     - method: set the time-stepping method
     - controller: select motion-based control (PDg, INV, OPS, EOM)
     - t_s: time step, dt [s]
     - t_end: maximum time [s]
     - t_save: sampling time step for history/viz [s]
     - flag_moving: activate moving desired target
     - flag_startpos: move starting position to initial joint (use inverse kinematics)

Author: Sebastian Heckers
Based on: Exercise 2b from 'Robot Dynamics, ETHZ
"""
import os
from Common import *
import RobotArmIRB120 as irb
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

## CONFIGURATION PARAMETERS
N_steps = 10000
t_s = 0.001
t_save = 0.01
t_end = 10.0
tau = np.zeros((6,1))
f_ext = np.zeros((6,1))
friction = 0.1*np.ones(6)
dq_0 = np.zeros(6)
q_0 = np.zeros(6)
method = "adaptive_rk4"
controller = "PDg"  # PDg INV OPS EOM
flag_friction = False
flag_gravity = True
flag_moving = True
flag_startpos = False

# goals
_move_r_des = lambda t_i:( irb.jointToPosition(q_0) + np.array([0, 0.2*np.sin(2*np.pi*t_i*0.2), 0.2*np.cos(2*np.pi*t_i*0.2)]) ).reshape((3,1))
q_des = np.ones(6)
r_des = _move_r_des(0.0) if flag_moving else irb.jointToPosition(q_0).reshape((3,1)) + np.ones((3,1))*0.1
eul_des = irb.rotMatToEulAngXYZ(irb.jointToRotMat(q_0)).reshape((3,1))
I_F_Ex = 10

print(f"EXAMPLE OF CONTROL FOR ABB ROBOT ARM\n")
print(f"    number of steps = {N_steps}")
print(f"          time step = {t_s} s")
print(f"         time total = {N_steps*t_s} s")
print(f"           time end = {t_end} s")
print(f"          force_ext = {[float(i[0]) for i in f_ext]}")
print(f"           friction = {friction} (enabled {flag_friction})")
print(f"            gravity = enabled {flag_gravity}")
print(f"             moving = enabled {flag_moving}")
print(f"      set start pos = enabled {flag_startpos}")
print(f" integration method = {method}")
print(f"  controller method = {controller}\n")

print(f"   q_des = {q_des.flatten()}")
print(f"   r_des = {r_des.flatten()}")
print(f" eul_des = {eul_des.flatten()}")
print(f"    F_Ex = {I_F_Ex}")

## LOADING KINEMATICS / EOM
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

## PREPARE SIMULATION STEP
step = irb.Integrators_Dynamics[method]

def reaction_collision_model(q,dq,Jpe=I_Jpe,x_wall=0.4,alpha=500.0):
    r_ = irb.jointToPosition(q)[0] - x_wall #+ 0.05
    v_ = (Jpe(*q) @ dq.reshape(6, 1))[0, 0]

    k = 180000.0
    d = 2000.0

    # activation = -1.0 / (1.0 + np.exp(-alpha * r_))
    # activation = 0.5 * (1.0 + np.tanh(alpha * r_))
    activation = np.logaddexp(0,alpha*r_)
    activation = np.maximum(0,activation)

    f_ = activation * (
        -k * r_ * np.array([1, 0, 0, 0, 0, 0])
        -d * v_ * np.array([1, 0, 0, 0, 0, 0])
    )
    f_ = np.minimum(0,f_) * 0.00001

    return f_.reshape((6, 1))

# plot collision model
x = np.linspace(-0.1,0.1,100)
fig, ax = plt.subplots()
for alp_ in [10,50,100,500,1000,2000]:
    ax.plot(x, [-1.0*reaction_collision_model(q_0,dq_0,x_wall=0.374-x_,alpha=alp_)[0,0] for x_ in x], label=f"alpha = {alp_}")
img_name = f"/bin/robotIRB120_control_collision_model.png"
fig.legend()
# ax.set_yscale('log')
plt.savefig(repo_root+img_name)
print(f"Saved plot: {img_name}")

## SIMULATION LOOP
q_i = irb.inverseKinematics(r_des, irb.jointToRotMat(q_0), q_0, 1e-6)[0] if flag_startpos else q_0
dq_i = dq_0
t_i = 0
t_next_save = t_save
it_final = N_steps
hist_ = [(q_i, dq_i, np.zeros(6), H(q_0,dq_0), E_kin(q_0,dq_0), E_pot(q_0), t_i )]
pbar = tqdm(total=N_steps)
t_start = perf_counter()
for i in range(N_steps):
    # set controller / desired goals
    cont = {"type": controller,
            "q_des": q_des,
            "r_des": r_des,
            "eul_des": eul_des,
            "I_F_Ex": I_F_Ex,
            "reaction_collision_model": reaction_collision_model}

    dynamics = irb.make_dynamics(f_ext, tau, eom, Jt, I_dJe, cont, flag_friction=flag_friction, friction=friction, flag_gravity=flag_gravity)
    q_i,dq_i,ddq,t_s = step(q_i,dq_i,t_s,dynamics)
    t_i += t_s

    # save current state
    if t_i >= t_next_save:
        hist_.append((q_i, dq_i, ddq, H(q_i,dq_i), E_kin(q_i,dq_i), E_pot(q_i), t_i))
        t_next_save += t_save

    # update progress bar
    pbar.set_postfix(t_i=f"{t_i:.3f}", dt=f"{t_s:.5g}")
    pbar.refresh()  
    pbar.update(1)

    # stop at time t_end
    if t_i >= t_end:
        it_final = i+1
        print(f"\nStopped simulation loop, reached final time.")
        break

    # move desired position
    if flag_moving:
        r_des = _move_r_des(t_i)

t_elapsed = perf_counter() - t_start
pbar.close()

print(f"TIMING:\n Simulation time: {t_elapsed:.3f} s\n Average time/step: {t_elapsed / it_final * 1000:.3f} ms")

dH = hist_[-1][3] - hist_[0][3]
print(f"RESULT: Energy H difference:\n total diff.: {dH:.3f}\n rel. diff.: {100*dH/hist_[0][3]:.3f}%\n rel. diff. per step: {100*(dH/hist_[0][3])/it_final:.3f}")
print(f"RESULT: time\n final time: {t_i}\n number of steps: {it_final}")

## SAVE HISTORY
data = np.array([
    np.concatenate([q, dq, ddq, [H, E_kin, E_pot, t]])
    for q, dq, ddq, H, E_kin, E_pot, t in hist_
])
header = "q1 q2 q3 q4 q5 q6 dq1 dq2 dq3 dq4 dq5 dq6 ddq1 ddq2 ddq3 ddq4 ddq5 ddq6 H E_kin E_pot t"
file_name = f"/bin/hist_robotIRB120_control_{controller}.txt"
np.savetxt(repo_root+file_name, data, header=header)
print(f"Saved history: {file_name}")

## PLOTTING
t = [h for _,_,_,_,_,_,h in hist_]
plotting_history_energy(t,hist_,img_name=f"/bin/robotIRB120_control_{controller}_energy.png")
plotting_history_joints(t,hist_,img_name=f"/bin/robotIRB120_control_{controller}_q.png")
plotting_history_pos(t,hist_,img_name=f"/bin/robotIRB120_control_{controller}_pos.png",x_des=np.array([list(irb.jointToPosition(q_des)) for h,_,_,_,_,_,_ in hist_]))

# colors = ['tab:blue', 'tab:orange', 'tab:green']
# labels = ['H', 'E_kin', 'E_pot']
# t = [h for _,_,_,_,_,_,h in hist_]

# # Energy
# fig, ax = plt.subplots()

# ax.plot(t, [h for _,_,_,h,_,_,_ in hist_], color='tab:red', label='H')
# ax.plot(t, [h for _,_,_,_,h,_,_ in hist_], color='tab:blue', label='E_kin')
# ax.plot(t, [h for _,_,_,_,_,h,_ in hist_], color='tab:green', label='E_pot')

# ax.set_title("Energy Diagram")
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Energy")
# ax.grid(True)
# ax.legend()

# img_name = f"/bin/robotIRB120_control_{controller}_energy.png"
# plt.savefig(repo_root+img_name)
# print(f"Saved plot: {img_name}")

# # Joint Angle
# fig, ax = plt.subplots()

# ax.plot(t, [list(h) for h,_,_,_,_,_,_ in hist_], label=[f"q_{i+1}" for i in range(6)])

# ax.set_title("Joint Angle Diagram")
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Angle [rad]")
# ax.grid(True)
# ax.legend()

# img_name = f"/bin/robotIRB120_control_{controller}_q.png"
# plt.savefig(repo_root+img_name)
# print(f"Saved plot: {img_name}")

# # Position
# fig, ax = plt.subplots()

# x_bot = np.array([list(irb.jointToPosition(h)) for h,_,_,_,_,_,_ in hist_])
# if controller == "PDg":
#     x_des = np.array([list(irb.jointToPosition(q_des)) for h,_,_,_,_,_,_ in hist_])
# else: # controller == "PD"
#     x_des = np.array([list(_move_r_des(t_i)) for t_i in t])
#     # x_des = np.array([list(r_des) for h,_,_,_,_,_,_ in hist_])
# for i in range(3):
#     ax.plot(t, x_bot[:,i], label=f"x_{i+1}", color=colors[i])
#     ax.plot(t, x_des[:,i], '--', label=f"x_des_{i+1}", color=colors[i])

# ax.set_title("End-effector Position Diagram")
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Position [m]")
# ax.grid(True)
# ax.legend()

# img_name = f"/bin/robotIRB120_control_{controller}_pos.png"
# plt.savefig(repo_root+img_name)
# print(f"Saved plot: {img_name}")

## Visualisation of Robot
irb.viz_hist([h for h,_,_,_,_,_,_ in hist_], dt=t_s)