import numpy as np
import numpy.fft as fft
from scipy import signal

# Main function

def cbf(volume, method = "fft", fps = 200, max_freq = 20, window = 25):
    """
    Convenience CBF function that invokes one of the three methods for
    computing CBF used in Quinn et al 2015, Science Translational Medicine.

    Parameters
    ----------
    volume : array, shape (F, H, W)
        A grayscale video of cilia, represented as a 3D volume
        with F frames, H rows, and W columns.
    method : string
        Selects the method used to compute CBF at each pixel (default: "fft").
        "fft" uses the raw FFT at each pixel, picking the dominant frequency.
        "psd" uses a simple periodogram to compute the power spectral density.
        "welch" uses the Welch sliding average method over a periodogram.
    fps : integer
        Framerate of the video, or frames per second (default: 200).
    max_freq : float
        Maximum allowable frequency; anything above this is clamped (default: 20).
    window : float
        Window size used in the Welch sliding average (default: 25).

    Returns
    -------
    heatmap : array, shape (H, W)
        A heatmap of dominant frequencies, one at each pixel location.
    """
    if method == "fft":
        return _cbf_fft(volume, fps, max_freq = max_freq)
    elif method == "psd":
        return _cbf_psd(volume, fps, max_freq = max_freq)
    elif method == "welch":
        return _cbf_welch(volume, fps, max_freq = max_freq, window = window)
    else:
        raise Error("Unrecognized method \"{}\" specified; only \"fft\", \"psd\", and \"welch\" supported.".format(method))

# CBF functions

def _cbf_fft(volume, fps, max_freq = 20):
    '''
    Calculates the ciliary beat frequency (CBF) of a given 3D volume. This
    function is fast but produces noisy results, as it only considers the
    instantaneous signal when extracting frequencies.

    Parameters
    ----------
    volume : array, shape (F, H, W)
        A 3D video volume with F frames, H rows, and W columns.
    fps : int
        Frames per second (capture framerate) of the video.
    max_freq : float
        Maximum allowed frequency; frequencies above this are clamped.

    Returns
    -------
    retval : array, shape (H, W)
        Heatmap of dominant frequencies at each spatial location (pixel).
    '''
    N = nextpow2(volume.shape[0])
    freq_bins = int(fps / 2) * np.linspace(0, 1, int(N / 2) + 1)
    retval = np.zeros(shape = (volume.shape[1], volume.shape[2]))

    # Subtract off the mean (always a good preprocessing step when FFT-ing).
    vol_0mean = volume - volume.mean(axis = 0)

    # Perform the FFT, take the frequency amplitudes.
    vol_fft = fft.fft(vol_0mean, n = N, axis = 0) / volume.shape[0]
    vol_abs = 2 * np.absolute(vol_fft[:int(N / 2) + 1])

    # Find the amplitude with the largest power.
    max_freq_indices = vol_abs.argmax(axis = 0)

    # Use the indices to extract the correct frequencies, and put them
    # at the correct locations in a heatmap.
    heatmap = freq_bins[max_freq_indices]

    # Maximal suppression.
    heatmap[heatmap > max_freq] = max_freq

    # All done.
    return heatmap

def cbf2(volume, fps, max_freq = 20):
    '''
    Calculates the ciliary beat frequency (CBF) of a given 3D volume.

    Parameters
    ----------
    volume : array, shape (F, H, W)
        A 3D video volume with F frames, H rows, and W columns.
    fps : int
        Frames per second (capture framerate) of the video.
    max_freq : float
        Maximum allowed frequency; frequencies above this are clamped.

    Returns
    -------
    retval : array, shape (H, W)
        Heatmap of dominant frequencies at each spatial location (pixel).
    '''
    N = nextpow2(volume.shape[0])

    # f, t, Sxx = signal.spectrogram(volume, fps, nfft = N,
    #     return_onesided = True, axis = 0, mode = "magnitude")
    f, Pxx = signal.periodogram(volume, fs = fps, nfft = N,
        return_onesided = True, axis = 0)

def cbf2(pixels, fps = 200, maximum_frequency = 20, maximum_frames = 128, detrend = True,
    window = 25.0, nfft = 128, noverlap = 127, pad_to = 256):
    """
    Calculates CBF using matplotlib's spectrogram to average frequencies across
    a temporal signal. If a CBF is computed which exceeds the maximum allowable
    frequency, the frequency with the next largest power is used, and so forth
    until a viable frequency is found. This prevents the buildup of power at 0
    or maximum_frequency for noisy signals.

    Parameters
    ----------
    pixels : array, shape (N, M)
        N instances of M-dimensional pixel trajectories.
    fps : integer
        Frames per second.
    maximum_frequency : integer
        Maximum frequency to consider. All frequencies over this threshold are reduced to it.
    maximum_frames : integer
        Maximum number of frames to consider.
    detrend : boolean
        If True, detrends the data using moving average.
    window : integer
        If detrend is True, window size used to perform convolution.
    nfft : integer
        See mlab.psd() documentation.
    noverlap : integer
        See mlab.psd() documentation.
    pad_to : integer
        See mlab.psd() documentation.

    Returns
    -------
    retval : array, shape (N,)
        List of CBFs for each pixel.
    """
    retval = np.zeros((pixels.shape[0]))
    i = 0
    w = np.ones(window) / float(window)
    for row in pixels:
        trajectory = pixels[i]
        if detrend is True:
            trajectory = pixels[i] - np.convolve(pixels[i], w, "same")
            trajectory = trajectory[(window / 2):np.size(trajectory) - (window / 2)]
        p, f = mlab.psd(trajectory[:maximum_frames], NFFT = nfft,
            detrend = mlab.detrend_linear, noverlap = noverlap, Fs = fps,
            pad_to = pad_to)
        idx = np.argmax(p)
        while f[idx] > maximum_frequency:
            p[idx] = 0.0
            idx = np.argmax(p)
        retval[i] = f[idx]
        i += 1
    return retval

# Utilities

def nextpow2(i):
    """
    Calculates the next even power of 2 which is greater than or equal to i.

    Parameters
    -----------
    i : integer
        The number which we want a power of 2 that is greater.

    Returns
    -------
    n : integer
        The next power of 2 such that n >= i.
    """
    n = 2
    while n < i:
        n *= 2
    return n
