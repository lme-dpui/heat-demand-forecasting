import numpy as np
import pywt
import matplotlib.pyplot as plt


def generate_wavelet(time_series, wavelet, num_freq, axis=-1):
    """Continuous wavelet transform of one or many signals.

    ``time_series`` may be a single 1-D signal or an n-D array of signals; the
    transform is taken along ``axis`` and a leading scale axis of length
    ``num_freq`` is prepended to the result. Passing a whole batch in one call
    lets PyWavelets vectorise the transform instead of looping in Python.
    """
    wavelet_transformed, freqs = pywt.cwt(time_series, range(1, num_freq + 1), wavelet, axis=axis)

    if wavelet_transformed.dtype == complex:
        wavelet_transformed = np.abs(wavelet_transformed) ** 2

    return wavelet_transformed, freqs


def plot_scalogram(scalogram, freqs, time_series=None, sampling_rate=24,
                   save_path=None, show=False, title=None):
    """Plot a wavelet scalogram, optionally above the signal it was computed from.

    Reproduces the two-panel figure used for the paper: the source signal on top
    and its ``|CWT|^2`` scalogram below, sharing the time axis. Matplotlib is
    imported lazily so that merely importing this module stays cheap.

    Parameters
    ----------
    scalogram : np.ndarray
        2-D scalogram ``[num_freq, time]`` as returned by :func:`generate_wavelet`
        for a single signal.
    freqs : array-like
        Frequencies for the vertical axis, as returned by :func:`generate_wavelet`.
    time_series : np.ndarray, optional
        The original 1-D signal; when given it is drawn in a panel above the
        scalogram.
    sampling_rate : int, optional
        Samples per time unit; the time axis is divided by this (24 -> days for
        hourly data). Defaults to 24.
    save_path : str or pathlib.Path, optional
        If given, the figure is written here at 600 dpi (parent dirs are created).
    show : bool, optional
        If True, display the figure interactively.
    title : str, optional
        Figure title.

    Returns
    -------
    (fig, axes)
        The Matplotlib figure and a tuple of its axes (signal axis first when a
        ``time_series`` is supplied, otherwise just the scalogram axis).
    """

    time = np.arange(scalogram.shape[-1]) / sampling_rate
    extent = [time[0], time[-1], freqs[0], freqs[-1]]

    if time_series is not None:
        fig, (ax_signal, ax_scalo) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 3]}, figsize=(6, 6))
        ax_signal.plot(np.arange(len(time_series)) / sampling_rate, time_series, c="cornflowerblue")
        ax_signal.set_ylabel("signal (a.u.)")
        axes = (ax_signal, ax_scalo)
    else:
        fig, ax_scalo = plt.subplots(figsize=(6, 4))
        axes = (ax_scalo,)

    im = ax_scalo.imshow(scalogram, cmap="magma", aspect="auto", extent=extent, origin="lower")
    fig.colorbar(im, ax=ax_scalo, orientation="horizontal", shrink=0.7, pad=0.2, label="amplitude (a.u.)")
    ax_scalo.set_xlabel("time (days)")
    ax_scalo.set_ylabel("frequency (Hz)")

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    if show:
        plt.show()

    return fig, axes


def create_image_representation(time_series, encoding_schema, num_freq, method="wavelet", axis=-1):
    image = None
    if method == "wavelet":
        image, freqs = generate_wavelet(time_series, encoding_schema.wavelet_function, num_freq, axis=axis)

    return image
