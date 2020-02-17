import numpy as np
#import cvapi.cv2api as cv2
import cv2
import scipy.signal

def optical_flow(video):
    """
    Computes optical flow for each pair of frames in the video.

    Parameters
    ----------
    video : array, shape (F, H, W)
        NumPy video array. F frames by H rows and W columns of pixel intensities.

    Returns
    -------
    X : array, shape (F - 1, H, W)
    Y : array, shape (F - 1, H, W)
        The X and Y components of the optical flow, respectively.
    """
    X = np.zeros(shape = (video.shape[0] - 1, video.shape[1], video.shape[2]))
    Y = np.zeros(shape = (video.shape[0] - 1, video.shape[1], video.shape[2]))
    for i in range(1, np.size(video, axis = 0)):
        prev_frame = video[i - 1]
        curr_frame = video[i]

        # Returns optical flow with dimensions (rows, cols, 2)
        #opt = cv2.optical_flow_fb(prev_frame, curr_frame, poly_n = 9, poly_sigma = 2.0, iterations = 100)
        #opt = cv2.optical_flow_fb(prev_frame, curr_frame, None, 0.5, 3, 30, 10, 7, 1.5, 0)
        opt = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, 0.5, 3, 30, 10, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        X[i - 1] = np.copy(opt[:, :, 0]) # X-component
        Y[i - 1] = np.copy(opt[:, :, 1]) # Y-component
    return [X, Y]

def invariants(X, Y):
    """
    Computes the image derivatives of the optical flow vectors so that
    we can compute some interesting differential image invariants.

    Parameters
    ----------
    X : array, shape (N, M)
        Optical flow vectors in the X direction for a single frame.
    Y : array, shape (N, M)
        Optical flow vectors in the Y direction for a single frame.

    Returns
    -------
    xu : array, shape (N, M)
        X-derivative of the X-component.
    yu : array, shape (N, M)
        X-derivative of the Y-component.
    xv : array, shape (N, M)
        Y-derivative of the X-component.
    yv : array, shape (N, M)
        Y-derivative of the Y-component.
    """

    # Set up the filter.
    sigmaBlur = 1
    gBlurSize = 2 * np.around(2.5 * sigmaBlur) + 1
    grid = np.mgrid[1:gBlurSize + 1] - np.around((gBlurSize + 1) / 2)
    g_filt = np.exp(-(grid ** 2) / (2 * (sigmaBlur ** 2)))
    g_filt /= np.sum(g_filt)
    dxg_filt = (-grid / (sigmaBlur ** 2)) * g_filt
    
    # Compute the derivatives.
    xu = scipy.signal.sepfir2d(X, dxg_filt, g_filt)
    yu = scipy.signal.sepfir2d(X, g_filt, dxg_filt)
    xv = scipy.signal.sepfir2d(Y, dxg_filt, g_filt)
    yv = scipy.signal.sepfir2d(Y, g_filt, dxg_filt)

    return [xu, yu, xv, yv]

def curl(X, Y):
    """
    Computes curl (rotation) for a single frame.

    Parameters
    ----------
    X : array, shape (N, M)
    Y : array, shape (N, M)
        X and Y components of the optical flow for a single frame.

    Returns
    -------
    retval : array, shape (N, M)
        Curl vectors for this frame.
    """
    _, yu, xv, _ = invariants(X, Y)
    return xv - yu

def deformation(X, Y):
    """
    Computes deformation (biaxial shear) for a single frame.

    Parameters
    ----------
    X : array, shape (N, M)
    Y : array, shape (N, M)
        X and Y components of the optical flow for a single frame.

    Returns
    -------
    retX : array, shape (N, M)
    retY : array, shape (N, M)
        X and Y components of the deformation vectors for this frame.
    """
    xu, yu, xv, yv = invariants(X, Y)
    return [xu - yv, xv + yu]
