import numpy as np
import numpy.fft as fft
from scipy import signal

# Main function

def cbf(volume, method = "fft", fps = 200, max_freq = 20):
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

    Returns
    -------
    heatmap : array, shape (H, W)
        A heatmap of dominant frequencies, one at each pixel location.
    """
    N = nextpow2(volume.shape[0])
    if method == "fft":
        maps = _cbf_fft(volume, fps, N, max_freq = max_freq)
    elif method == "psd":
        maps = _cbf_psd(volume, fps, N, max_freq = max_freq)
    elif method == "welch":
        maps = _cbf_welch(volume, fps, N, max_freq = max_freq)
    else:
        raise Error("Unrecognized method \"{}\" specified; " \
            "only \"fft\", \"psd\", and \"welch\" supported.".format(method))


# CBF functions

def _cbf_fft(volume, fps, N, max_freq = 20):
    """
    Calculates the ciliary beat frequency (CBF) of a given 3D volume. This
    function is fast but produces noisy results, as it only considers the
    instantaneous signal when extracting frequencies.

    Parameters
    ----------
    volume : array, shape (F, H, W)
        A 3D video volume with F frames, H rows, and W columns.
    fps : int
        Frames per second (capture framerate) of the video.
    N : int
        Number of bins for the discrete transform.
    max_freq : float
        Maximum allowed frequency; frequencies above this are clamped.

    Returns
    -------
    retval : array, shape (H, W)
        Heatmap of dominant frequencies at each spatial location (pixel).
    """
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

def _cbf_psd(volume, fps, N, max_freq = 20):
    """
    Calculates the ciliary beat frequency (CBF) of a given 3D volume. This
    function computes a simple periodogram of frequencies in a given signal.
    """
    N = nextpow2(volume.shape[0])
    f, Pxx = signal.periodogram(volume, fs = fps, nfft = N,
        return_onesided = True, axis = 0)
    max_freq_indices = Pxx.argmax(axis = 0)
    heatmap = f[max_freq_indices]
    heatmap[heatmap > max_freq] = max_freq
    return heatmap

def _cbf_welch(volume, fps, N, max_freq = 20):
    """
    Calculates the ciliary beat frequency (CBF) of a given 3D volume. This
    function uses the Welch algorithm to build a periodogram that is smoothed
    using a windowed average, generating a robust estimation of the spectral
    density of the signal.
    """
    N = nextpow2(volume.shape[0])
    f, Pxx = signal.welch(volume, fs = fps, window = "hann", nfft = N,
        return_onesided = True, axis = 0)
    max_freq_indices = Pxx.argmax(axis = 0)
    heatmap = f[max_freq_indices]
    heatmap[heatmap > max_freq] = max_freq
    return heatmap

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
