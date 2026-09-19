"""
JOINT TO TRANSFORMATION FUNCTIONS in SYMPY
Functions for the joints/generalized coordinates and their dynamics. It
contains homogenous transformation matrices (jointToTransformXY), from the
inertial to end-effector frame, and the end-effector position and orientation.
It uses Sympy to make it symbolical.

Author: Sebastian Heckers
Based on: jointToTransform.py
"""
import sympy as sp

def getTransformI0():
    """
    Input: void
    Output: T_I0 - homogenous transformation matrix from frame 0 to frame I
    """
    return sp.eye(4)

def getTransform6E():
    """
    Input: void
    Output: T_6E - homogenous transformation matrix from frame E to frame 6
    """
    return sp.eye(4)

def jointToTransform01(q):
    """
    Input: q - joint angles
    Output: T_01 - homogenous transformation matrix from frame 1 to frame 0
    """
    p = q[0]
    r01 = sp.Matrix([0, 0, 0.145])#.reshape((3,1))
    C01 = sp.Matrix([[sp.cos(p), -sp.sin(p), 0],
                     [sp.sin(p),  sp.cos(p), 0],
                     [        0,          0, 1]])
    
    T_01 = sp.Matrix.vstack(
        sp.Matrix.hstack(C01, r01),
        sp.Matrix([[0, 0, 0, 1]])
    )
    return T_01
    
def jointToTransform12(q):
    """
    Input: q - joint angles
    Output: T_12 - homogenous transformation matrix from frame 2 to frame 1
    """
    p = q[1]
    r12 = sp.Matrix([0, 0, 0.145])#.reshape((3,1))
    C12 = sp.Matrix([[ sp.cos(p), 0, sp.sin(p)],
                     [         0, 1,         0],
                     [-sp.sin(p), 0, sp.cos(p)]])
    
    T_12 = sp.Matrix.vstack(
        sp.Matrix.hstack(C12, r12),
        sp.Matrix([[0, 0, 0, 1]])
    )
    return T_12

def jointToTransform23(q):
    """
    Input: q - joint angles
    Output: T_23 - homogenous transformation matrix from frame 3 to frame 2
    """
    p = q[2]
    r23 = sp.Matrix([0, 0, 0.27])#.reshape((3,1))
    C23 = sp.Matrix([[ sp.cos(p), 0, sp.sin(p)],
                     [         0, 1,         0],
                     [-sp.sin(p), 0, sp.cos(p)]])
    
    T_23 = sp.Matrix.vstack(
        sp.Matrix.hstack(C23, r23),
        sp.Matrix([[0, 0, 0, 1]])
    )
    return T_23

def jointToTransform34(q):
    """
    Input: q - joint angles
    Output: T_34 - homogenous transformation matrix from frame 4 to frame 3
    """
    p = q[3]
    r34 = sp.Matrix([0.134, 0, 0.07])#.reshape((3,1))
    C34 = sp.Matrix([[ 1,         0,         0],
                     [ 0, sp.cos(p),-sp.sin(p)],
                     [ 0, sp.sin(p), sp.cos(p)]])
    
    T_34 = sp.Matrix.vstack(
        sp.Matrix.hstack(C34, r34),
        sp.Matrix([[0, 0, 0, 1]])
    )
    return T_34

def jointToTransform45(q):
    """
    Input: q - joint angles
    Output: T_45 - homogenous transformation matrix from frame 5 to frame 4
    """
    p = q[4]
    r45 = sp.Matrix([0.168, 0, 0])#.reshape((3,1))
    C45 = sp.Matrix([[ sp.cos(p), 0, sp.sin(p)],
                     [         0, 1,         0],
                     [-sp.sin(p), 0, sp.cos(p)]])
    
    T_45 = sp.Matrix.vstack(
        sp.Matrix.hstack(C45, r45),
        sp.Matrix([[0, 0, 0, 1]])
    )
    return T_45

def jointToTransform56(q):
    """
    Input: q - joint angles
    Output: T_56 - homogenous transformation matrix from frame 6 to frame 5
    """
    p = q[5]
    r56 = sp.Matrix([0.072, 0, 0])#.reshape((3,1))
    C56 = sp.Matrix([[ 1,         0,         0],
                     [ 0, sp.cos(p),-sp.sin(p)],
                     [ 0, sp.sin(p), sp.cos(p)]])
    
    T_56 = sp.Matrix.vstack(
        sp.Matrix.hstack(C56, r56),
        sp.Matrix([[0, 0, 0, 1]])
    )
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

    r_IE = TI0 @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T6E @ sp.Matrix([0,0,0,1]).T

    I_r_IE = r_IE[0:3]

    return I_r_IE

def jointToRotMat(q):
    """
    Input: q - joint angles
    Output: C_IE - rotation matrix from the end-effector frame E to the inertial frame I
    """
    Cx = lambda p: sp.Matrix( 
        [[ 1,         0,         0],
         [ 0, sp.cos(p),-sp.sin(p)],
         [ 0, sp.sin(p), sp.cos(p)]] )
    Cy = lambda p: sp.Matrix( 
        [[ sp.cos(p), 0, sp.sin(p)],
         [         0, 1,         0],
         [-sp.sin(p), 0, sp.cos(p)]] )
    Cz = lambda p: sp.Matrix( 
        [[sp.cos(p), -sp.sin(p), 0],
         [sp.sin(p),  sp.cos(p), 0],
         [        0,          0, 1]] )

    C_IE = Cz(q[0]) @ Cy(q[1]) @ Cy(q[2]) @ Cx(q[3]) @ Cy(q[4]) @ Cx(q[5])

    return C_IE