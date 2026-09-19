"""
ROTATION FUNCTIONS
Helper functions to handle various rotation representations.

Author: Sebastian Heckers
Based on: Exercises from 'Robot Dynamics', ETHZ
"""
import numpy as np


def rotMatToRotVec(C):
    """
    Input: C - rotation matrix
    Output: phi - rotational vector describing rotation of C
    """
    theta = np.arccos(
        np.clip((C[0,0] + C[1,1] + C[2,2] - 1.0) / 2, -1.0, 1.0)
    )

    if np.isclose(theta, 0.0, atol=1e-8):
        phi = np.zeros((3,1))
    else:
        n = 1 / (2*np.sin(theta)) * np.array([
            C[2,1] - C[1,2],
            C[0,2] - C[2,0],
            C[1,0] - C[0,1]
        ]).reshape(3,1)

        phi = theta*n

    return phi

def eulAngXYZToRotMat(angles):
    """
    Input: angles - XYZ Euler angles
    Output: C - rotation matrix
    """
    angles = list(angles.flatten())
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

    C = Cx(angles[0]) @ Cy(angles[1]) @ Cz(angles[2])

    return C

def rotMatToEulAngXYZ(C):
    """
    Input: C - rotation matrix
    Output: angles - XYZ Euler angles
    """
    y = np.arcsin(np.clip(C[0, 2], -1.0, 1.0))
    x = np.arctan2(-C[1, 2], C[2, 2])
    z = np.arctan2(-C[0, 1], C[0, 0])
    angles = np.array([x, y, z])  # Radiant
    return angles
