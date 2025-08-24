
import sys
import argparse
import configparser
import pickle
import numpy as np
import plotly.graph_objects as go

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import utils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--filepath", help="dandi filepath", type=str,
                        default="../../data/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--bin_size", help="bin size (secs)", type=float,
                        default=0.02)
    parser.add_argument("--skip_sqrt_transform",
                        help="sqrt transform spike counts", action="store_true")
    parser.add_argument("--save_filename_pattern", help="save filename pattern", type=str,
                        default="../../results/binned_spikes_dandisetID{:s}_binSize{:.2f}_skipLogTrans{:d}.npz")
    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    filepath = args.filepath
    bin_size = args.bin_size
    skip_sqrt_transform = args.skip_sqrt_transform
    save_filename = args.save_filename_pattern.format(dandiset_ID, bin_size,
                                                      skip_sqrt_transform)

    #%%
    # Download data
    # ^^^^^^^^^^^^^
    # with DandiAPIClient() as client:
    #     asset = client.get_dandiset(dandiset_ID, "draft").get_asset_by_path(filepath)
    #     s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)

    # io = NWBHDF5IO(s3_path, mode="r", driver="ros3")

    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()

    # n_clusters
    n_clusters = units_df.shape[0]

    # continuous spikes times
    continuous_spikes_times = [None for n in range(n_clusters)]
    for n in range(n_clusters):
        continuous_spikes_times[n] = units_df.iloc[n]['spike_times']

    binned_spikes, bin_edges = utils.bin_spike_times(
        spike_times=continuous_spikes_times, bin_size=bin_size)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    if not skip_sqrt_transform:
        binned_spikes = np.sqrt(binned_spikes + 0.5)

    np.savez(save_filename, bin_size=bin_size,
             slip_sqrt_transform=skip_sqrt_transform,
             binned_spikes=binned_spikes,
             bin_centers=bin_centers)


if __name__ == "__main__":
    main(sys.argv)
