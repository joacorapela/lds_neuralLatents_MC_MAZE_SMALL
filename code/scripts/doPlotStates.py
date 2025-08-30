import sys
import pickle
import argparse
import configparser
import pynwb
import numpy as np

import plotUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_time", help="earliest time to plot", type=float,
                        default=-np.inf)
    parser.add_argument("--to_time", help="latest time to plot", type=float,
                        default=np.inf)
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--filepath_pattern", help="dandi filepath", type=str,
                        default="../../data/{:s}/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--events_names_to_plot",
                        help="names of events to plot", type=str,
                        default="start_time,target_on_time,go_cue_time,move_onset_time,stop_time")
    parser.add_argument("--events_linetypes_to_plot",
                        help="linetypes of events to plot", type=str,
                        default="dot,dash,dashdot,longdash,solid")
    parser.add_argument("--filtered_data_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=49497641)
                        # default=26118000)
                        # default=59816097)
    parser.add_argument("--variable", type=str, default="state",
                        help="variable to plot: state")
    parser.add_argument("--color_pattern_filtered", type=str,
                        default="rgba(255,0,0,{:f})",
                        help="color pattern for filtered data")
    parser.add_argument("--cb_alpha", type=float, default=0.3,
                        help="transparency factor for confidence bound")
    parser.add_argument("--filtered_data_filenames_pattern", type=str,
                        default="../../results/{:08d}_filtered.{:s}",
                        help="filtered_data filename pattern")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/{:08d}_{:s}_filtered_from{:.02f}_to{:.02f}.{:s}")

    args = parser.parse_args()

    from_time = args.from_time
    to_time = args.to_time
    dandiset_ID = args.dandiset_ID
    filepath_pattern = args.filepath_pattern
    events_names_to_plot = args.events_names_to_plot.split(",")
    events_linetypes_to_plot = args.events_linetypes_to_plot.split(",")
    filtered_data_number = args.filtered_data_number
    variable = args.variable
    cb_alpha = args.cb_alpha
    filtered_data_filenames_pattern = \
        args.filtered_data_filenames_pattern
    fig_filename_pattern = args.fig_filename_pattern

    filepath = filepath_pattern.format(dandiset_ID)
    with pynwb.NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    filtered_data_filename = \
        filtered_data_filenames_pattern.format(filtered_data_number, "pickle")
    with open(filtered_data_filename, "rb") as f:
        filtered_data = pickle.load(f)

    filtered_metadata_filename = \
        filtered_data_filenames_pattern.format(filtered_data_number, "ini")
    filtered_metadata = configparser.ConfigParser()
    filtered_metadata.read(filtered_metadata_filename)

    bin_centers = filtered_data["bin_centers"]
    first_index = np.where(bin_centers >= from_time)[0][0]
    last_index = np.where(bin_centers <= to_time)[0][-1]
    to_plot_slice = slice(first_index, last_index)
    bin_centers_to_plot = bin_centers[to_plot_slice]
    means_to_plot = filtered_data["xnn"][:,:,to_plot_slice]
    covs_to_plot = filtered_data["Pnn"][:,:,to_plot_slice]
    if variable == "state":
        fig = plotUtils.plot_latents(
            means=means_to_plot,
            covs=covs_to_plot,
            bin_centers=bin_centers_to_plot,
            trials_df=trials_df,
            events_names_to_plot=events_names_to_plot,
            events_linetypes_to_plot=events_linetypes_to_plot,
            cb_alpha=cb_alpha,
        )
    else:
        raise ValueError("variable={:s} is invalid.")

    fig.update_layout(
        title=f'Log-Likelihood: {filtered_data["logLike"].squeeze()}')
    fig.write_image(fig_filename_pattern.format(filtered_data_number, variable,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(filtered_data_number, variable,
                                               from_time, to_time, "html"))
    fig.show()
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
