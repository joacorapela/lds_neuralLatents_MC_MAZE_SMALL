import sys
import pickle
import numpy as np
import argparse
import configparser
import plotly.graph_objects as go
import pynwb

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--filepath", help="dandi filepath", type=str,
                        default="../../data/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--events_names_to_plot",
                        help="names of events to plot", type=str,
                        default="start_time,target_on_time,go_cue_time,move_onset_time,stop_time")
    parser.add_argument("--events_linetypes_to_plot",
                        help="linetypes of events to plot", type=str,
                        default="dot,dash,dashdot,longdash,solid")
    parser.add_argument("--filtered_data_number", type=int,
                        help="number corresponding to filtered results filename",
                        default=26118000)
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
                        default="../../figures/{:08d}_{:s}_filtered.{:s}")

    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    filepath = args.filepath
    events_names_to_plot = args.events_names_to_plot.split(",")
    events_linetypes_to_plot = args.events_linetypes_to_plot.split(",")
    filtered_data_number = args.filtered_data_number
    variable = args.variable
    color_pattern_filtered = args.color_pattern_filtered
    cb_alpha = args.cb_alpha
    filtered_data_filenames_pattern = \
        args.filtered_data_filenames_pattern
    fig_filename_pattern = args.fig_filename_pattern

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

    N = filtered_data["xnn"].shape[2]
    bin_centers = filtered_data["bin_centers"]
    if variable == "state":
        fig = go.Figure()
        n_states = filtered_data["xnn"].shape[0]
        for i in range(n_states):
            filter_means = filtered_data["xnn"][i, 0, :]
            filter_stds = np.sqrt(filtered_data["Pnn"][i, i, :])
            filter_ci_upper = filter_means + 1.96*filter_stds
            filter_ci_lower = filter_means - 1.96*filter_stds

            trace = go.Scatter(
                x=bin_centers, y=filter_means,
                mode="lines+markers",
                marker={"color": color_pattern_filtered.format(1.0)},
                name=f"filtered_{i}",
                showlegend=True,
                legendgroup=f"filtered_{i}",
            )
            trace_cb = go.Scatter(
                x=np.concatenate([bin_centers, bin_centers[::-1]]),
                y=np.concatenate([filter_ci_upper, filter_ci_lower[::-1]]),
                fill="toself",
                fillcolor=color_pattern_filtered.format(cb_alpha),
                line=dict(color=color_pattern_filtered.format(0.0)),
                showlegend=False,
                legendgroup=f"filtered_{i}",
            )
            fig.add_trace(trace)
            fig.add_trace(trace_cb)

        n_trials = trials_df.shape[0]
        for r in range(n_trials):
            for e, event_name in enumerate(events_names_to_plot):
                event_linetype_to_plot = events_linetypes_to_plot[e]
                fig.add_vline(x=trials_df.iloc[r][event_name],
                              line_dash=event_linetype_to_plot)
        fig.update_xaxes(title="Time (sec)")
        fig.update_yaxes(title="Latent Value")

    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))

    fig.update_layout(title=f'Log-Likelihood: {filtered_data["logLike"].squeeze()}')
    fig.write_image(fig_filename_pattern.format(filtered_data_number, variable, "png"))
    fig.write_html(fig_filename_pattern.format(filtered_data_number, variable, "html"))
    fig.show()
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
