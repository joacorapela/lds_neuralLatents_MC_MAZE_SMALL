
import numpy as np

def bin_spike_times(spike_times, bin_size=0.1, t_max=None):
    """
    Bin spike times into spike count matrix.

    Parameters
    ----------
    spike_times : list of lists
        spike_times[n] = list of spike times (floats) for neuron n
    bin_size : float
        Bin width in seconds (default 0.1 = 100 ms)
    t_max : float or None
        End time for binning. If None, set to max spike time across all neurons.

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Binned spike counts.
    bin_edges : ndarray
        The bin edges used.
    """
    n_neurons = len(spike_times)
    if t_max is None:
        t_max = max(max(st) for st in spike_times if len(st) > 0)

    # Compute bin edges
    bin_edges = np.arange(0, t_max + bin_size, bin_size)
    n_bins = len(bin_edges) - 1

    # Allocate matrix
    spike_counts = np.zeros((n_neurons, n_bins), dtype=int)

    # Bin spikes per neuron
    for n, spikes in enumerate(spike_times):
        spike_counts[n], _ = np.histogram(spikes, bins=bin_edges)

    return spike_counts, bin_edges

def matrix_to_string(A):
    return ";".join([",".join(map(str, row)) for row in A])

def vector_to_string(v):
    return ",".join(map(str, v))

