"""
PSEUDO INVERSE MATRIX
The Moore-Penrose pseudo-inverse for non-square matrices.

Author: Sebastian Heckers
Based on: Exercise 1c from 'Robot Dynamics', ETHZ
"""
import numpy as np


def pseudoInverseMat(A, lam):
    """
    Returns the pseudo inverse of the input according to the Moore-Penrose formula.

    Input: A - nxn matrix, lam - damping factor
    Output: pinvA - pseudo-inverse of A
    """
    (m,n) = A.shape

    if m >= n and np.linalg.matrix_rank(A) == n:
        pinvA = np.linalg.inv(A.T @ A + lam**2 * np.eye(n)) @ A.T
    else:
        pinvA = A.T @ np.linalg.inv(A @ A.T + lam**2 * np.eye(m))

    return pinvA