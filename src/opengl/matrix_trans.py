"""
Adapted from: https://github.com/BryceQing/OPENCV_AR
"""

import numpy as np
import cv2

def extrinsic_to_model_view(RVEC, TVEC, R_vector = True):
    """[Get modelview matrix from RVEC and TVEC]

    Arguments:
        RVEC {[vector]} -- [Rotation vector]
        TVEC {[vector]} -- [Translation vector]
    """
    
    if RVEC.shape[0] > 1:
        RVEC = RVEC[:1]
    
    R, _ = cv2.Rodrigues(RVEC)
    
    Rx = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    if TVEC.shape[0] > 1:
        TVEC = TVEC[:1]

    TVEC = TVEC.flatten().reshape((3, 1))

    transform_matrix = Rx @ np.hstack((R, TVEC))
    M = np.eye(4)
    M[:3, :] = transform_matrix
    return M.T.flatten()


def intrinsic_to_project(MTX, width, height, near_plane=0.01, far_plane=100.0):
    """[Get ]

    Arguments:
        MTX {[np.array]} -- [The camera instrinsic matrix that you get from calibrating your chessboard]
        width {[float]} -- [width of viewport]]
        height {[float]} -- [height of viewport]

    Keyword Arguments:
        near_plane {float} -- [near_plane] (default: {0.01})
        far_plane {float} -- [far plane] (default: {100.0})

    Returns:
        [np.array] -- [1 dim array of project matrix]
    """
    P = np.zeros(shape=(4, 4), dtype=np.float32)
    
    fx, fy = MTX[0, 0], MTX[1, 1]
    cx, cy = MTX[0, 2], MTX[1, 2]
    
    
    P[0, 0] = 2 * fx / width
    P[1, 1] = 2 * fy / height
    P[2, 0] = 1 - 2 * cx / width
    P[2, 1] = 2 * cy / height - 1
    P[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    P[2, 3] = -1.0
    P[3, 2] = - (2 * far_plane * near_plane) / (far_plane - near_plane)

    return P.flatten()
