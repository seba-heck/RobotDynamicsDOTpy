"""
DIFFERENTIAL KINEMATICS FUNCTIONS
The functions jointToPosJac and jointToRotJac return the jacobian.

Author: Sebastian Heckers
Based on: Exercise 1b from 'Robot Dynamics', ETHZ
"""
import numpy as np
from .jointToTransform import *


def jointToPosJac(q):
    """
    Returns translation Jacobian which maps joint velocities to 
    end-effector linear velocities in frame I.

    Input: q - joint angles
    Output: J_p - jacobian of end-effector translation
    """
    # get relative homogeneous transformation matrices
    T_I0 = getTransformI0()
    T_01 = jointToTransform01(q)
    T_12 = jointToTransform12(q)
    T_23 = jointToTransform23(q)
    T_34 = jointToTransform34(q)
    T_45 = jointToTransform45(q)
    T_56 = jointToTransform56(q)
    T_6E = getTransform6E()
    # homogeneous transformation matrices from frame k to inertial frame I
    T_I1 = T_I0 @ T_01
    T_I2 = T_I1 @ T_12
    T_I3 = T_I2 @ T_23
    T_I4 = T_I3 @ T_34
    T_I5 = T_I4 @ T_45
    T_I6 = T_I5 @ T_56
    # unit vector of link rotation
    n_1 = T_I1 @ np.array([0,0,1,0])
    n_2 = T_I2 @ np.array([0,1,0,0])
    n_3 = T_I3 @ np.array([0,1,0,0])
    n_4 = T_I4 @ np.array([1,0,0,0])
    n_5 = T_I5 @ np.array([0,1,0,0])
    n_6 = T_I6 @ np.array([1,0,0,0])
    # position vectors from each homogeneous transformation
    I_r_I1 = T_I1[0:3,3]
    I_r_I2 = T_I2[0:3,3]
    I_r_I3 = T_I3[0:3,3]
    I_r_I4 = T_I4[0:3,3]
    I_r_I5 = T_I5[0:3,3]
    I_r_I6 = T_I6[0:3,3]
    # compute end-effector position vector
    I_r_IE = jointToPosition(q)
    # assemble jacobian matrix
    J_p = np.array( [
        np.cross( n_1[0:3], (I_r_IE - I_r_I1) ),
        np.cross( n_2[0:3], (I_r_IE - I_r_I2) ),
        np.cross( n_3[0:3], (I_r_IE - I_r_I3) ),
        np.cross( n_4[0:3], (I_r_IE - I_r_I4) ),
        np.cross( n_5[0:3], (I_r_IE - I_r_I5) ),
        np.cross( n_6[0:3], (I_r_IE - I_r_I6) )
    ] )

    return J_p.T

def jointToRotJac(q):
    """
    Returns rotational jacobian which maps joint velocities to 
    end-effector linear velocities in frame I.

    Input: q - joint angles
    Output: J_R - jacobian of end-effector orientation
    """
    # get relative homogeneous transformation matrices
    T_I0 = getTransformI0()
    T_01 = jointToTransform01(q)
    T_12 = jointToTransform12(q)
    T_23 = jointToTransform23(q)
    T_34 = jointToTransform34(q)
    T_45 = jointToTransform45(q)
    T_56 = jointToTransform56(q)
    T_6E = getTransform6E()
    # homogeneous transformation matrices from frame k to inertial frame I
    T_I1 = T_I0 @ T_01
    T_I2 = T_I1 @ T_12
    T_I3 = T_I2 @ T_23
    T_I4 = T_I3 @ T_34
    T_I5 = T_I4 @ T_45
    T_I6 = T_I5 @ T_56
    # unit vector of link rotation
    n_1 = T_I1 @ np.array([0,0,1,0])
    n_2 = T_I2 @ np.array([0,1,0,0])
    n_3 = T_I3 @ np.array([0,1,0,0])
    n_4 = T_I4 @ np.array([1,0,0,0])
    n_5 = T_I5 @ np.array([0,1,0,0])
    n_6 = T_I6 @ np.array([1,0,0,0])
    # assemble jacobian matrix
    J_R = np.array([n_1[0:3],n_2[0:3],n_3[0:3],n_4[0:3],n_5[0:3],n_6[0:3]])

    return J_R.T