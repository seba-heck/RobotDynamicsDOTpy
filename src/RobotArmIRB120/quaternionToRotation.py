"""
FUNCTIONS FOR QUATERNIONS
Functions to handle quaternions representing the orientation.

Author: Sebastian Heckers
Based on: Exercises (1a) from 'Robot Dynamics', ETHZ
"""
import numpy as np

def rotMatToQuat(R):
    """
    Input: R - rotation matrix
    Output: q - quaternion [w x y z]
    """
    q = 0.5 * np.array([
        np.sqrt(np.trace(R)+1),
        np.sign(R[2,1] - R[1,2]) * np.sqrt(R[0,0] - R[1,1] - R[2,2] + 1),
        np.sign(R[0,2] - R[2,0]) * np.sqrt(R[1,1] - R[0,0] - R[2,2] + 1),
        np.sign(R[1,0] - R[0,1]) * np.sqrt(R[2,2] - R[0,0] - R[1,1] + 1)
    ])
    return q

def quatToRotMat(q):
    """
    Input: q - quaternion
    Output: R - rotation matrix
    """
    R = [[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3]],
         [2*q[0]*q[3] + 2*q[1]*q[2], q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
         [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[0]*q[1] + 2*q[2]*q[3], q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]]
    return np.array(R)

def quatMult(q_AB, q_BC):
    """
    Input: q_AB, q_BC - two quaternions for multiplication
    Output: q_AC - resulting quaternion
    """
    w,x,y,z = q_AB
    M = np.array([
        [w, -z, y],
        [z, w, -x],
        [-y, x, w]
    ])
    tmp = np.hstack((q_AB.reshape(4,1), np.vstack((-q_AB[1:4].reshape(1,3), M)) ))
    q_AC = tmp @ q_BC
    return q_AC

def rotVecWithQuat(q_BA, A_r):
    """
    Rotate vector/point around quaternion.
    Input: q_BA - orientation quaternion, A_r - coordinate to map
    Output: B_r - coordinates in target frame
    """
    R_BA = quatToRotMat(q_BA)
    B_r = R_BA @ A_r.T
    return B_r