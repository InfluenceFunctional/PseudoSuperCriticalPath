import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

import argparse
import os
from typing import Callable, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx} \DeclareSIUnit{\atm}{atm} \DeclareSIUnit{\cal}{cal}"
})


def yield_valid_rows(log_file: str) -> Iterable[tuple[int, int]]:
    start_index = None
    final_index = None
    with open(log_file, "r") as file:
        for line_index, line in enumerate(file):
            if line.strip().startswith("Step"):
                assert start_index is None
                start_index = line_index + 1
            if line.strip().startswith("Loop time"):
                assert start_index is not None
                assert final_index is None
                final_index = line_index - 1
                yield start_index, final_index
                start_index = None
                final_index = None


def compose_row_function(log_file: str, minimization: bool) -> Callable[[int], bool]:
    ranges = []
    for index, (start, end) in enumerate(yield_valid_rows(log_file)):
        if minimization and index == 0:
            continue
        ranges.append([start, end])
    # Include header.
    if ranges == []:
        raise ValueError("No valid rows found")
        print("Invalid screen.log file")
    else:
        ranges[0][0] -= 1

    # Remove overlaps.
    for i in range(1, len(ranges)):
        ranges[1][0] += 1

    def skip_row(row: int) -> bool:
        for s, e in ranges:
            if s <= row <= e:
                return False
        return True

    return skip_row


def read_molecule_wise_temperature(moltemp_file: str):
    in_timestep_data = False
    frames = {}
    current_timestep = None
    with open(moltemp_file, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            split_line = line.split()
            if len(split_line) == 2:
                assert not in_timestep_data
                assert current_timestep is None
                number_frame = int(split_line[0])
                number_molecules = int(split_line[1])
                assert number_frame not in frames
                frames[number_frame] = np.zeros((number_molecules, 3))
                current_timestep = number_frame
                in_timestep_data = True
            elif len(split_line) == 4:
                assert in_timestep_data
                molecule_index = int(split_line[0]) - 1
                temp = float(split_line[1])
                kecom = float(split_line[2])
                internal = float(split_line[3])
                frames[current_timestep][molecule_index] = [temp, kecom, internal]
                if molecule_index == frames[current_timestep].shape[0] - 1:
                    in_timestep_data = False
                    current_timestep = None
            else:
                raise RuntimeError
    first_frame = frames[list(frames.keys())[0]]
    assert all(np.all(frame.shape == first_frame.shape) for frame in frames.values())
    return frames, first_frame.shape[0]


def plot(data, variable, label, unit_string, ramp_index, filename_base):
    plt.figure()
    plt.plot(data["Time"] / 1000.0 / 1000.0, data[variable])
    plt.xlabel(r"time $t / \unit{\nano\second}$")
    plt.ylabel(label)
    mean = np.mean(data[variable][ramp_index:])
    plt.axhline(mean, color="black", linestyle="dashed",
                label=f"average: $\\SI{{{mean:.2f}}}{{\\{unit_string}}}$")
    plt.axvline(data["Time"][ramp_index] / 1000.0 / 1000.0, color="black", linestyle="dotted")
    plt.legend()
    plt.savefig(f"{filename_base}_with_ramp.pdf", bbox_inches="tight")

    plt.figure()
    plt.plot(data["Time"][ramp_index:] / 1000.0 / 1000.0, data[variable][ramp_index:])
    plt.xlabel(r"time $t / \unit{\nano\second}$")
    plt.ylabel(label)
    plt.axhline(mean, color="black", linestyle="dashed",
                label=f"average: $\\SI{{{mean:.2f}}}{{{unit_string}}}$")
    plt.legend()
    plt.savefig(f"{filename_base}.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze LAMMPS log files.")
    parser.add_argument("log", help="LAMMPS log file")
    parser.add_argument("moltemp", help="molecular temperature file")
    parser.add_argument("output", help="output directory")
    parser.add_argument("--ramp_time", help="ramp time in fs", type=int, default=2 * 1000 * 1000)
    parser.add_argument("--minimization", help="skip the first part of the log file", action="store_true")
    args = parser.parse_args()

    log = args.log
    output_directory = args.output
    moltemp = args.moltemp
    ramp_time = args.ramp_time
    minimization = args.minimization

    # noinspection PyTypeChecker
    data = pd.read_csv(log, skiprows=compose_row_function(log, minimization), sep=r"\s+")
    # As soon as this fails, I have to think about how to handle the moltemp file.
    assert np.all(data["Time"] == data["Step"])
    moltemp_data, number_molecules = read_molecule_wise_temperature(moltemp)
    ramp_index = np.argmax(data["Time"] >= ramp_time)
    print("RAMP INDEX: ", ramp_index)

    try:
        os.mkdir(output_directory)
    except FileExistsError:
        pass

    plot(data, "Temp", r"temperature $T / \unit{\kelvin}$", "\\kelvin", ramp_index,
         f"{output_directory}/temperature")
    plot(data, "Volume", r"volume $V / \unit{\angstrom^3}$", "\\angstrom^3",
         ramp_index, f"{output_directory}/volume")
    plot(data, "Press", r"pressure $P / \unit{\atm}$", "\\atm", ramp_index,
         f"{output_directory}/pressure")
    plot(data, "KinEng", r"kinetic energy $K / \unit{\kilo\cal\per\mole}$",
         "\\kilo\\cal\\per\\mole", ramp_index, f"{output_directory}/kinetic_energy")
    plot(data, "PotEng", r"potential energy $E / \unit{\kilo\cal\per\mole}$",
         "\\kilo\\cal\\per\\mole", ramp_index, f"{output_directory}/potential_energy")
    plot(data, "E_pair", r"pair energy $E_{\mathrm{pair}} / \unit{\kilo\cal\per\mole}$",
         "\\kilo\\cal\\per\\mole", ramp_index, f"{output_directory}/pair_energy")
    plot(data, "E_mol", r"molecular energy $E_{\mathrm{mol}} / \unit{\kilo\cal\per\mole}$",
         "\\kilo\\cal\\per\\mole", ramp_index, f"{output_directory}/molecular_energy")

    temp_fig = plt.figure()
    temp_ax = temp_fig.subplots()
    temp_fig_without_ramp = plt.figure()
    temp_ax_without_ramp = temp_fig_without_ramp.subplots()
    kecom_fig = plt.figure()
    kecom_ax = kecom_fig.subplots()
    kecom_fig_without_ramp = plt.figure()
    kecom_ax_without_ramp = kecom_fig_without_ramp.subplots()
    internal_fig = plt.figure()
    internal_ax = internal_fig.subplots()
    internal_fig_without_ramp = plt.figure()
    internal_ax_without_ramp = internal_fig_without_ramp.subplots()
    steps = np.array([key for key in moltemp_data.keys()])
    ramp_index = np.argmax(steps >= ramp_time)
    for molecule_index in range(number_molecules):
        temp_values = [moltemp_data[step][molecule_index][0] for step in steps]
        kecom_values = [moltemp_data[step][molecule_index][1] for step in steps]
        internal_values = [moltemp_data[step][molecule_index][2] for step in steps]
        temp_ax.plot(steps, temp_values, alpha=0.5)
        temp_ax_without_ramp.plot(steps[ramp_index:], temp_values[ramp_index:], alpha=0.2)
        kecom_ax.plot(steps, kecom_values, alpha=0.5)
        kecom_ax_without_ramp.plot(steps[ramp_index:], kecom_values[ramp_index:], alpha=0.2)
        internal_ax.plot(steps, internal_values, alpha=0.5)
        internal_ax_without_ramp.plot(steps[ramp_index:], internal_values[ramp_index:], alpha=0.2)

    temp_mean = [np.mean([moltemp_data[step][molecule_index][0] for molecule_index in range(number_molecules)])
                 for step in steps]
    kecom_mean = [np.mean([moltemp_data[step][molecule_index][1] for molecule_index in range(number_molecules)])
                  for step in steps]
    internal_mean = [np.mean([moltemp_data[step][molecule_index][2] for molecule_index in range(number_molecules)])
                     for step in steps]

    temp_ax.plot(steps, temp_mean, color="black")
    temp_ax_without_ramp.plot(steps[ramp_index:], temp_mean[ramp_index:], color="black")
    kecom_ax.plot(steps, kecom_mean, color="black")
    kecom_ax_without_ramp.plot(steps[ramp_index:], kecom_mean[ramp_index:], color="black")
    internal_ax.plot(steps, internal_mean, color="black")
    internal_ax_without_ramp.plot(steps[ramp_index:], internal_mean[ramp_index:], color="black")

    temp_ax.set_xlabel(r"time $t / \unit{\nano\second}$")
    temp_ax_without_ramp.set_xlabel(r"time $t / \unit{\nano\second}$")
    kecom_ax.set_xlabel(r"time $t / \unit{\nano\second}$")
    kecom_ax_without_ramp.set_xlabel(r"time $t / \unit{\nano\second}$")
    internal_ax.set_xlabel(r"time $t / \unit{\nano\second}$")
    internal_ax_without_ramp.set_xlabel(r"time $t / \unit{\nano\second}$")

    temp_ax.set_ylabel(r"molecule-wise temperature $T_\mathrm{mol} / \unit{\kelvin}$")
    temp_ax_without_ramp.set_ylabel(r"molecule-wise temperature $T_\mathrm{mol} / \unit{\kelvin}$")
    kecom_ax.set_ylabel(r"molecule-wise kinetic energy $K_\mathrm{mol} / \unit{\kilo\cal\per\mole}$")
    kecom_ax_without_ramp.set_ylabel(r"molecule-wise kinetic energy $K_\mathrm{mol} / \unit{\kilo\cal\per\mole}$")
    internal_ax.set_ylabel(r"molecule-wise internal kinetic energy $K_\mathrm{mol,int} / \unit{\kilo\cal\per\mole}$")
    internal_ax_without_ramp.set_ylabel(
        r"molecule-wise internal kinetic energy $K_\mathrm{mol,int} / \unit{\kilo\cal\per\mole}$")

    temp_fig.savefig(f"{output_directory}/molecule_wise_temperature_with_ramp.png", bbox_inches="tight")
    temp_fig_without_ramp.savefig(f"{output_directory}/molecule_wise_temperature.png",
                                  bbox_inches="tight")
    kecom_fig.savefig(f"{output_directory}/molecule_wise_kecom_with_ramp.png", bbox_inches="tight")
    kecom_fig_without_ramp.savefig(f"{output_directory}/molecule_wise_kecom.png", bbox_inches="tight")
    internal_fig.savefig(f"{output_directory}/molecule_wise_internal_with_ramp.png", bbox_inches="tight")
    internal_fig_without_ramp.savefig(f"{output_directory}/molecule_wise_internal.png", bbox_inches="tight")

    temp_fig.clear()
    temp_fig_without_ramp.clear()
    kecom_fig.clear()
    kecom_fig_without_ramp.clear()
    internal_fig.clear()
    internal_fig_without_ramp.clear()


if __name__ == '__main__':
    main()




plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx} \DeclareSIUnit{\atm}{atm} \DeclareSIUnit{\cal}{cal}"
})
plt.rcParams.update(plt.rcParamsDefault)


def plot_run_thermo(lmbd):
    """
    Utility function to plot the thermo data for a particular run
    :param lmbd:
    :return:
    """
    directory = Path("stage_one_fixed/runs_at_each_lambda/")

    lambda_dirs = os.listdir(directory)
    lambda_dirs = [dir for dir in lambda_dirs if "lambda_" in dir]
    number_molecules = np.loadtxt("npt_simulations/fluid/T_300/tmp.out", skiprows=3, max_rows=1, dtype=int)[1]
    run_directory = Path(f"{directory}/lambda_{lmbd:.2g}")
    log_file = run_directory.joinpath("screen.log")
    data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")

    from plotly.subplots import make_subplots
    rows = 3
    cols = 4
    keys_to_plot = ['E_pair', 'E_vdw', 'Press', 'E_coul', 'E_long', 'E_mol', 'c_lj', 'c_coul', 'c_gauss', 'KinEng',
                    'PotEng', 'TotEng']
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=keys_to_plot)

    for ind, k1 in enumerate(keys_to_plot):
        row = ind // cols + 1
        col = ind % cols + 1
        fig.add_scatter(x=data['Time'], y=data[k1], showlegend=False, row=row, col=col)

    fig.show()


def analyze_gen_runs():
    run_directory = Path(r'D:\crystal_datasets\pscp\stage_three\generate_restart_files_lambda')
    log_file = run_directory.joinpath("screen.log")
    data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")

    from plotly.subplots import make_subplots

    rows = 2
    cols = 5
    keys_to_plot = ['E_pair', 'Volume', 'Temp', 'Press', 'KinEng', 'PotEng', 'TotEng', 'E_mol', 'E_pair_grad',
                    'P_grad']
    data['E_pair_grad'] = np.gradient(data['E_pair'])
    data['P_grad'] = np.gradient(data['Press'])
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=keys_to_plot)

    for ind, k1 in enumerate(keys_to_plot):
        row = ind // cols + 1
        col = ind % cols + 1
        fig.add_scatter(x=data['Time'], y=data[k1], showlegend=False, row=row, col=col)

    #fig.show(renderer='browser')

    '''
    try to predict local overlaps
    '''

    energy = data['PotEng']

    num_lambda_steps = 100
    #lambda_steps = optimize_lambda_spacing3(np.array(energy), num_lambda_steps=num_lambda_steps, buffer=50, maxiter=200)
    lambda_steps = optimize_lambda_spacing(np.array(energy[::10]), num_lambda_steps=num_lambda_steps, buffer=10,
                                           maxiter=1000)


def extract_lambda_spacing(num_lambda_steps, num_restart_paths, gen_restart_path: str, buffer=10):
    run_directory = Path(gen_restart_path)
    log_file = run_directory.joinpath("screen.log")
    data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")
    energy = data['PotEng']

    subsample_ratio = len(energy) // num_restart_paths

    lambda_steps = optimize_lambda_spacing(np.array(energy)[::subsample_ratio],
                                           num_lambda_steps=num_lambda_steps,
                                           buffer=buffer,
                                           maxiter=1000,
                                           showfigs=False)

    return lambda_steps


def spacing_analysis_fig(data, energy, lambda_steps, lambdas, buffer):
    lambda_energies, lambda_widths, nn_overlaps, overlaps = (
        batch_compute_overlaps(buffer, energy, lambda_steps))
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['Energy and lambda points', 'Overlaps', 'Nearest-neighbor overlaps'])
    fig.add_scatter(x=data['Time'],
                    y=energy,
                    name='Energy',
                    row=1, col=1
                    )
    fig.add_scatter(x=data['Time'][lambda_steps],
                    y=lambda_energies,
                    error_y=dict(type='data', array=lambda_widths, visible=True),
                    showlegend=True,
                    name='Binned Energy',
                    row=1, col=1
                    )
    fig.add_heatmap(z=np.log(overlaps),
                    row=1, col=2
                    )
    fig.add_scatter(x=lambdas[lambda_steps][1:],
                    y=nn_overlaps,
                    showlegend=True,
                    name='Nearest-Neighbor Overlap',
                    row=1, col=3
                    )
    fig.show(renderer='browser')


def optimize_lambda_spacing(energy, num_lambda_steps, buffer, maxiter=200, showfigs=True):
    all_steps = np.arange(0, len(energy))
    lambda_energies, lambda_widths, full_nn_overlaps, full_overlaps = batch_compute_overlaps(buffer, energy, all_steps)

    current_steps = np.linspace(0, len(energy) - 1, num_lambda_steps).astype(int)

    lambda_energies, lambda_widths, nn_overlaps, overlaps = (
        batch_compute_overlaps(buffer, energy, current_steps))

    iter = 0
    tolerance = 1
    converged = False
    overlaps_record = []
    states_record = []
    while not converged and iter < maxiter:
        if nn_overlaps.min() < nn_overlaps.mean() * 0.5:
            tolerance *= 0.975
        target_overlap = nn_overlaps.mean() * tolerance

        greedy_steps = []
        i0 = 0
        greedy_steps.append(i0)
        for s_ind in range(num_lambda_steps - 1):
            state_overlaps = full_overlaps[i0]
            valid_options = state_overlaps >= target_overlap
            pick_state = np.amax(np.where(valid_options))  # pick the furthest valid state
            assert pick_state > i0, "No valid overlaps after this given state"
            greedy_steps.append(pick_state)
            i0 = pick_state
            if i0 == len(energy) - 1:
                #print("Ran out of states!")
                tolerance *= 1.01

                # fill in with randoms
                while ((num_lambda_steps) - len(greedy_steps)) > 0:
                    randint = np.random.randint(len(energy) - 2)
                    if randint not in greedy_steps:
                        greedy_steps.append(randint)
                        greedy_steps.sort()

                break

        lambda_energies, lambda_widths, nn_overlaps, overlaps = (
            batch_compute_overlaps(buffer, energy, current_steps))

        overlaps_record.append(nn_overlaps)

        greedy_steps[-1] = len(energy) - 1
        states_record.append(greedy_steps)
        current_steps = np.array(greedy_steps)

        iter += 1

    min_record = np.array([ov.min() for ov in overlaps_record])
    mean_record = np.array([ov.mean() for ov in overlaps_record])
    if showfigs:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_scatter(y=min_record, name='min')
        fig.add_scatter(y=mean_record, name='mean')
        fig.show(renderer='browser')
        go.Figure(go.Scatter(y=min_record * mean_record, name='combo score')).show(renderer='browser')

    best_greedy_state = np.asarray(states_record[np.argmax(min_record)])
    mins = []
    exp_states = []
    opt_exponents = np.concatenate([np.linspace(0.1, 1.2, 20), np.ones(1)])
    for exp in opt_exponents:
        exp_state = np.copy(best_greedy_state)
        exp_state = ((exp_state / len(energy)) ** exp) * len(energy)  # rescale slightly
        exp_state = exp_state.astype(int)
        lambda_energies, lambda_widths, nn_overlaps, overlaps = (
            batch_compute_overlaps(buffer, energy, exp_state))
        mins.append(nn_overlaps.min())
        exp_states.append(exp_state)

    best_state = exp_states[np.argmax(np.array(mins))]

    if showfigs:
        run_directory = Path(r'D:\crystal_datasets\pscp\stage_three\generate_restart_files_lambda')
        log_file = run_directory.joinpath("screen.log")
        data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")

        lambdas = np.linspace(0, 1, len(energy))
        spacing_analysis_fig(data, energy, best_state, lambdas, buffer=buffer)

        fig = go.Figure(
            go.Scatter(x=np.linspace(0, 1, len(best_state)), y=best_state / best_state[-1], mode='lines+markers'))
        fig.add_scatter(x=np.linspace(0, 1, len(best_state)), y=np.linspace(0, 1, len(best_state))).show(
            renderer='browser')

    return best_state


def batch_compute_overlaps(buffer, energy, steps_to_sample):
    lambda_energies = np.zeros(len(steps_to_sample))
    lambda_widths = np.zeros(len(steps_to_sample))
    lambda_steps = steps_to_sample
    for ind in range(len(steps_to_sample)):
        s1 = max(lambda_steps[ind] - buffer, 0)
        s3 = min(lambda_steps[ind] + buffer, len(energy) - 2)

        step_inds = np.arange(s1, s3)

        lambda_energies[ind] = np.mean(energy[step_inds])
        lambda_widths[ind] = np.std(energy[step_inds])
    overlaps = np.exp(
        -(lambda_energies[:, None] - lambda_energies[None, :]) ** 2 / (
                lambda_widths[:, None] ** 2 + lambda_widths[None, :] ** 2))
    nn_overlaps = np.array(
        [overlaps[ind, ind - 1] for ind in range(1, len(steps_to_sample))]
    )
    return lambda_energies, lambda_widths, nn_overlaps, overlaps


if __name__ == '__main__':
    analyze_gen_runs()
