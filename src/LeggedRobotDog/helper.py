import os,sys
import numpy as np
from .generateEOM import get_T,get_J


# End-effector position & orientation and velocity
def get_position_EE(I_T_EE):        
    p_EE = np.array([
            I_T_EE[0, 3],
            I_T_EE[2, 3],
            np.arctan2(I_T_EE[0, 2], I_T_EE[0, 0])
        ]).reshape((3,1))
    return p_EE

def get_position_EE_from_model(model,data,q,dq):
    q = np.asarray(q, dtype=float).reshape(-1)
    dq = np.asarray(dq, dtype=float).reshape(-1)
    _,_,_,I_T_EE = get_T(model,data,q,dq)
    return get_position_EE(I_T_EE)

def get_velocity_EE(I_J_EE, dq):
    w_EE = I_J_EE @ dq
    return w_EE

def get_velocity_EE_from_model(model,data,q,dq):
    q = np.asarray(q, dtype=float).reshape(-1)
    dq = np.asarray(dq, dtype=float).reshape(-1)
    _,_,_,I_J_EE = get_J(model,data,q)
    return get_velocity_EE(I_J_EE,dq)