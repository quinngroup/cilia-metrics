import numpy as np

def raster_scan(video):
    """
    Converts a 3D video volume (with frames on the first axis) to a 2D
    matrix (with pixels as rows, and frames as columns).

    Parameters
    ----------
    video : array, shape (F, H, W)
        A NumPy array with F frames, H rows, and W columns.

    Returns
    -------
    matrix : array, shape (H * W, F)
        Raster-scanned (row-stacked) matrix where rows are pixels.
    """
    # First, move the frames to the last dimension.
    video = np.swapaxes(video, 0, 2)

    # Second, swap the spatial dimensions.
    video = np.swapaxes(video, 0, 1)

    # Third, just reshape.
    n_pixels = video.shape[0] * video.shape[1]
    n_frames = video.shape[2]
    matrix = video.reshape((n_pixels, n_frames))
    return matrix
