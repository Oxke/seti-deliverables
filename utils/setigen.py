#!/usr/bin/env python
from collections.abc import Iterable
from typing import Optional, Union, Sequence
import numpy as np
import astropy.units as u
from time import perf_counter as pc
from typing import Any, Tuple
import multiprocessing
import setigen_patch as stg
import astropy.units as u

## LOOK AT THE noise_only_MeerKAT_5b.ipynb notebook for clarifications
T_rx = lambda nu: 10.6 * u.K + 0.633 * (nu - 8.41 * u.GHz) * (u.K / u.GHz)
T_sky = lambda nu: 568 * u.K * (nu / u.GHz) ** -1.13  # galactic center
T_sys = lambda nu: T_rx(nu) + T_sky(nu)


def power(nu, frame):
    k_B = 1.38e-23 * (u.J / u.K)
    R = 50 * u.Ohm
    P = k_B * T_sys(nu) * frame.df * u.Hz
    P = P.to(u.yW)
    N = (frame.df * u.Hz * frame.dt * u.s).to(u.dimensionless_unscaled)
    return P, N


def add_signal(frame, mean_freq, dr, snr, width):
    f_start = mean_freq - dr * frame.data.shape[0] * frame.dt * u.s / 2
    return frame.add_signal(
        stg.constant_path(f_start=f_start, drift_rate=dr),
        stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=width),
        stg.constant_bp_profile(level=1),
    )


def make_and_save(
    *args: Tuple[Any, ...],
    signal_per_file: int = 1,
    num_samples: int = 2**22,
    n_ints: int = 16,
    df: u.Quantity = 1 * u.Hz,
    dt: u.Quantity = 1 * u.s,
    hi: u.Quantity = 10 * u.GHz,
    signal_info_dir: str = "",
    output_dir: str = "",
    time_it: bool = False,
) -> str:
    """
    Create a synthetic data frame with injected signals and save it to an HDF5
    file.

    Parameters:
    -----------
    *args : Tuple
        A tuple of signal parameter arrays (e.g., midF, dr, snr, width), one per signal.
    signal_per_file : int, optional
        Number of signals to include in the frame. Default is 1.
    num_samples : int, optional
        Number of frequency channels in the frame. Default is 2**22.
    n_ints : int, optional
        Number of time integrations in the frame. Default is 16.
    df : Quantity, optional
        Frequency resolution per channel. Default is 1 Hz.
    dt : Quantity, optional
        Time resolution per integration. Default is 1 s.
    hi : Quantity, optional
        Top frequency (fch1) of the frame. Default is 10 GHz.
    signal_info_dir : str, optional
        Directory to store additional signal info (e.g., parameters). Default is empty string.
    output_dir : str, optional
        Directory to save the resulting HDF5 file. Default is empty string.
    time_it : bool, optional
        Whether to print timing information for each step. Default is False.

    Returns:
    --------
    str
        The name of the saved file.
    """
    if signal_per_file == 1:
        filename = "_".join(f"{arg.value:.4g}" for arg in args) + ".h5"
    else:
        filename = str(hash(str(args))) + ".h5"
        with open(signal_info_dir + filename + ".npy", "wb") as f:
            np.save(f, args)

    print(f"{filename} running on process {multiprocessing.current_process().name}")

    if time_it:
        print("Creating Frame...", end=" ")
        start = pc()

    frame = stg.Frame(
        fchans=num_samples,
        tchans=n_ints,
        df=df,
        dt=dt,
        fch1=hi,
        dtype=np.float32,
    )

    if time_it:
        print(f"{pc() - start:.3f} s")
        print("Calculating Noise parameters... ", end=" ")
        step = pc()

    P, N = power(np.atleast_1d(args[0]).mean(), frame)  # args[0] is mid frequency array

    if time_it:
        print(f"{pc() - step:.3f} s")
        print("Adding Noise... ", end=" ")
        step = pc()

    noise = frame.add_noise(
        k=(N / 2).value, theta=(2 * P / N).value, noise_type="gamma"
    )

    for i in range(signal_per_file):
        if time_it:
            print(f"{pc() - step:.3f} s")
            print("Adding Signal... ", end=" ")
            step = pc()

        signal = add_signal(frame, *[arg[i] for arg in args])

    if time_it:
        print(f"{pc() - step:.3f} s")
        print("Saving File... ", end=" ")
        step = pc()

    frame.save_h5(output_dir + filename)

    if time_it:
        print(f"{pc() - step:.3f} s")
        print(f"\nTOTAL : {pc() - start:.3f} s")

    return filename


MIDF = 8.3 * u.GHz, 15.4 * u.GHz  # MeerKAT
DR = 0.01 * (u.Hz / u.s), 5 * (u.Hz / u.s)
SNR = 10, 1000
WIDTH = 1 * u.Hz, 20 * u.Hz


def _make_and_save_random(
    signal_per_file: int = 1,
    seed: Optional[int] = None,
    time_it: bool = False,
    midf: Optional[Union[u.Quantity, Sequence[u.Quantity]]] = MIDF,
    dr: Optional[Union[u.Quantity, Sequence[u.Quantity]]] = DR,
    snr: Optional[
        Union[float, Sequence[float], u.Quantity, Sequence[u.Quantity]]
    ] = SNR,
    width: Optional[Union[u.Quantity, Sequence[u.Quantity]]] = WIDTH,
    evenly_spaced: bool = False,
    adjust_snr: bool = False,
    **kwargs,
) -> None:
    """
    Generate and save a synthetic signal dataset with randomized parameters.
    This is the inner function and should not be called directly. Instead, use
    the `make_and_save_random` function (and same parameters, in addition to
    those for i/o and threading)

    Parameters:
    -----------
    signal_per_file : int, optional
        Number of signals to generate per file. Default is 1.
    seed : int, optional
        Seed for the random number generator. Default is None.
    time_it : bool, optional
        Whether to time the generation and saving process. Default is False.
    midf : quantity or sequence of quantity, optional
        (middle) frequency to use. if a sequence of length 2 is given, it's interpreted as a range [min, max].
        if a single quantity is given, it is broadcasted to all signals.
        default is [8.3 GHz, 15.4 GHz] (MeerKAT band 5b)
    dr : quantity or sequence of quantity, optional
        absolute value of drift rate(s) to use. if a sequence of length 2 is given, it's interpreted as a range [min, max].
        if a single quantity is given, it is broadcasted to all signals.
        The sign of the drift rate is randomly picked afterwards.
        default is [0.01, 5]
    snr : float, Quantity, or sequence, optional
        Signal-to-noise ratio(s). If a sequence of length 2 is given, it is interpreted as a range [min, max].
        If a single value is given, it is broadcasted to all signals.
        default is [10, 1000]
    width : Quantity or sequence of Quantity, optional
        Signal width(s). If a sequence of length 2 is given, it is interpreted as a range [min, max].
        If a single quantity is given, it is broadcasted to all signals.
    evenly_spaced : bool, optional
        If True, space mid-frequencies evenly across the frequency band.
        Otherwise, choose them randomly around a central frequency. Default is False.
    **kwargs : dict
        Additional keyword arguments passed to `make_and_save`.

    Returns:
    --------
    None
        The function creates and saves the synthetic signal file(s), and prints the output filename.
    """
    rng = np.random.default_rng(seed=seed)
    if isinstance(midf, Sequence):
        assert len(midf) == 2, "Only accepted [min, max] as astropy quantities"
        midf = midf[0] + rng.random() * (midf[1] - midf[0])

    if isinstance(dr, Sequence):
        assert len(dr) == 2, "Only accepted [min, max] as astropy quantities"
        dr = dr[0] + rng.random(signal_per_file) * (dr[1] - dr[0])
    else:
        dr = dr * np.ones(signal_per_file)

    if isinstance(snr, Sequence):
        assert len(snr) == 2, "Only accepted [min, max] as astropy quantities"
        lsnr_lo, lsnr_hi = np.log10(snr[0]), np.log10(snr[1])
        snr = (
            10 ** (lsnr_lo + rng.random(signal_per_file) * (lsnr_hi - lsnr_lo))
            * u.dimensionless_unscaled
        )
    else:
        snr = snr * np.ones(signal_per_file)

    if isinstance(width, Sequence):
        assert len(width) == 2, "Only accepted [min, max] as astropy quantities"
        width = width[0] + rng.random(signal_per_file) * (width[1] - width[0])
    else:
        width = width * np.ones(signal_per_file)

    if adjust_snr:
        snr /= np.sqrt(width.to(u.Hz).value)

    df = 1 * u.Hz
    n_samples = 2**22
    n_ints = 16
    obstime = 300 * u.s
    dt = obstime / n_ints
    f_hi = midf + df * (n_samples // 2)

    if evenly_spaced:
        midF = np.linspace(
            midf - 0.495 * n_samples * df,
            midf + 0.495 * n_samples * df,
            signal_per_file,
        )
    else:
        midF = midf + (rng.random(signal_per_file) - 0.5) * n_samples * df

    output_filename = make_and_save(
        midF,
        dr,
        snr,
        width,
        signal_per_file=signal_per_file,
        num_samples=n_samples,
        n_ints=n_ints,
        df=df,
        dt=dt,
        hi=f_hi,
        time_it=time_it,
        **kwargs,
    )
    print(
        f"Process {multiprocessing.current_process().name} made file {output_filename}"
    )


def make_and_save_random(
    num_cpus=10, max_iterations=22_831, target=_make_and_save_random, **kwargs
):
    """
    Run a target function in parallel across multiple processes, up to a fixed
    number of iterations.

    This function spawns `num_cpus` worker processes, each repeatedly invoking
    the given `target` function until the total number of invocations across all
    processes reaches `max_iterations`.

    A shared counter (protected by a lock) ensures that no more than
    `max_iterations` total calls to `target` are made, regardless of how many
    processes are running in parallel.

    Parameters:
    - num_cpus (int): Number of parallel worker processes to spawn. Default is
      10.
    - max_iterations (int): Total number of times to invoke `target` across all
      processes. Default is 22,831.  target (callable): A function to be
      executed repeatedly by each worker process. This function must accept all
      keyword arguments passed via `**kwargs`.
    - **kwargs: Arbitrary keyword arguments passed to the `target` function on
      each invocation.

    Notes:
    - The function `target` is called as `target(**kwargs)` by each worker.
    - Iteration counting is handled via a shared `multiprocessing.Value` to
      synchronize across processes.
    - All processes are joined before the function returns.
    """

    def _worker(counter, lock, max_iterations, kwargs):
        while True:
            with lock:
                if counter.value >= max_iterations:
                    break
                counter.value += 1
                current_iteration = counter.value
            target(**kwargs)

    # Shared value and lock for synchronization
    counter = multiprocessing.Value("i", 0)  # shared integer
    lock = multiprocessing.Lock()

    processes = []
    for _ in range(num_cpus):
        p = multiprocessing.Process(
            target=_worker, args=(counter, lock, max_iterations, kwargs)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes completed.")
