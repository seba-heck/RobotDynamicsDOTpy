"""
JOINT TO TRANSFORMATION FUNCTIONS
Functions for the joints/generalized coordinates and their dynamics. It
contains homogenous transformation matrices (jointToTransformXY), from the
inertial to end-effector frame, and the end-effector position and orientation.

Author: Sebastian Heckers
Based on: Exercise 1a from 'Robot Dynamics', ETHZ
"""
import numpy as np
from .quaternionToRotation import rotMatToQuat

def getTransformI0():
    """
    Input: void
    Output: T_I0 - homogenous transformation matrix from frame 0 to frame I
    """
    return np.eye(4)

def getTransform6E():
    """
    Input: void
    Output: T_6E - homogenous transformation matrix from frame E to frame 6
    """
    return np.eye(4)

def jointToTransform01(q):
    """
    Input: q - joint angles
    Output: T_01 - homogenous transformation matrix from frame 1 to frame 0
    """
    p = q[0]
    r01 = np.array([0, 0, 0.145]).reshape((3,1))
    C01 = np.array( [[np.cos(p), -np.sin(p), 0],
                     [np.sin(p),  np.cos(p), 0],
                     [        0,          0, 1]])
    
    T_01 = np.block([[C01, r01], [np.zeros((1,3)), 1.0] ])

    return T_01
    
def jointToTransform12(q):
    """
    Input: q - joint angles
    Output: T_12 - homogenous transformation matrix from frame 2 to frame 1
    """
    p = q[1]
    r12 = np.array([0, 0, 0.145]).reshape((3,1))
    C12 = np.array( [[ np.cos(p), 0, np.sin(p)],
                     [         0, 1,         0],
                     [-np.sin(p), 0, np.cos(p)]])
    
    T_12 = np.block([[C12, r12], [np.zeros((1,3)), 1.0] ])

    return T_12

def jointToTransform23(q):
    """
    Input: q - joint angles
    Output: T_23 - homogenous transformation matrix from frame 3 to frame 2
    """
    p = q[2]
    r23 = np.array([0, 0, 0.27]).reshape((3,1))
    C23 = np.array( [[ np.cos(p), 0, np.sin(p)],
                     [         0, 1,         0],
                     [-np.sin(p), 0, np.cos(p)]])
    
    T_23 = np.block([[C23, r23], [np.zeros((1,3)), 1.0] ])

    return T_23

def jointToTransform34(q):
    """
    Input: q - joint angles
    Output: T_34 - homogenous transformation matrix from frame 4 to frame 3
    """
    p = q[3]
    r34 = np.array([0.134, 0, 0.07]).reshape((3,1))
    C34 = np.array( [[ 1,         0,         0],
                     [ 0, np.cos(p),-np.sin(p)],
                     [ 0, np.sin(p), np.cos(p)]])
    
    T_34 = np.block([[C34, r34], [np.zeros((1,3)), 1.0] ])

    return T_34

def jointToTransform45(q):
    """
    Input: q - joint angles
    Output: T_45 - homogenous transformation matrix from frame 5 to frame 4
    """
    p = q[4]
    r45 = np.array([0.168, 0, 0]).reshape((3,1))
    C45 = np.array( [[ np.cos(p), 0, np.sin(p)],
                     [         0, 1,         0],
                     [-np.sin(p), 0, np.cos(p)]])
    
    T_45 = np.block([[C45, r45], [np.zeros((1,3)), 1.0] ])

    return T_45

def jointToTransform56(q):
    """
    Input: q - joint angles
    Output: T_56 - homogenous transformation matrix from frame 6 to frame 5
    """
    p = q[5]
    r56 = np.array([0.072, 0, 0]).reshape((3,1))
    C56 = np.array( [[ 1,         0,         0],
                     [ 0, np.cos(p),-np.sin(p)],
                     [ 0, np.sin(p), np.cos(p)]])
    
    T_56 = np.block([[C56, r56], [np.zeros((1,3)), 1.0] ])

    return T_56

def jointToPosition(q):
    """
    Input: q - joint angles
    Output: I_r_IE - position of end-effector w.r.t. inertial frame
    """
    TI0 = getTransformI0()
    T01 = jointToTransform01(q)
    T12 = jointToTransform12(q)
    T23 = jointToTransform23(q)
    T34 = jointToTransform34(q)
    T45 = jointToTransform45(q)
    T56 = jointToTransform56(q)
    T6E = getTransform6E()

    r_IE = TI0 @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T6E @ np.array([0,0,0,1]).T

    I_r_IE = r_IE[0:3]

    return I_r_IE

def jointToRotMat(q):
    """
    Input: q - joint angles
    Output: C_IE - rotation matrix from the end-effector frame E to the inertial frame I
    """
    Cx = lambda p: np.array( 
        [[ 1,         0,         0],
         [ 0, np.cos(p),-np.sin(p)],
         [ 0, np.sin(p), np.cos(p)]] )
    Cy = lambda p: np.array( 
        [[ np.cos(p), 0, np.sin(p)],
         [         0, 1,         0],
         [-np.sin(p), 0, np.cos(p)]] )
    Cz = lambda p: np.array( 
        [[np.cos(p), -np.sin(p), 0],
         [np.sin(p),  np.cos(p), 0],
         [        0,          0, 1]] )

    C_IE = Cz(q[0]) @ Cy(q[1]) @ Cy(q[2]) @ Cx(q[3]) @ Cy(q[4]) @ Cx(q[5])

    return C_IE
    
def jointToQuat(q):
    """
    Input: q - joint angles
    Output: quat_IE - quaternion of the orientation of the end-effector
    """
    c = jointToRotMat(q)
    quat_IE = rotMatToQuat(c)
    return quat_IE