import os
from pathlib import Path

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

from analyze import compose_row_function

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


def analyze_gen_runs(show_figs=False):
    head_directory = Path(r'D:\crystal_datasets\pscp\stage_two\generate_restart_files_lambda')
    os.chdir(head_directory)
    #dirs = os.listdir()
    #dirs = [dir for dir in dirs if 'well_width' in dir]
    dirs = [os.getcwd()]
    en_records = []
    for dir in dirs:
        run_directory = head_directory.joinpath(Path(dir))
        log_file = run_directory.joinpath("screen.log")
        data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")
        data['E_pair_grad'] = np.gradient(data['E_pair'])
        data['P_grad'] = np.gradient(data['Press'])
        en_records.append(data['E_pair'])

        if show_figs:
            make_run_thermo_fig(data, dir)

    # extract the run with the smoothest transition
    en_records = np.vstack(en_records)

    sigma = 100
    normed_ens = en_records / np.abs(np.mean(en_records, axis=1))[:, None]
    smoothed_ens = gaussian_filter1d(normed_ens, sigma=sigma)
    diffs = np.diff(smoothed_ens, axis=1)
    grads = np.diff(diffs, axis=1)

    fig = make_subplots(rows=1, cols=3, subplot_titles=['Energy', '1st deriv', '2nd deriv'])
    for ind in range(len(en_records)):
        fig.add_scatter(y=en_records[ind] / np.abs(en_records[ind].mean()), row=1, col=1, name=dirs[ind],
                        legendgroup=dirs[ind], showlegend=True)
        fig.add_scatter(y=diffs[ind], row=1, col=2, name=dirs[ind], legendgroup=dirs[ind], showlegend=False)
        fig.add_scatter(y=grads[ind], row=1, col=3, name=dirs[ind], legendgroup=dirs[ind], showlegend=False)
    fig.show(renderer='browser')

    well_width = [0.1, 0.5, 1.0, 2.0, 4.0]
    well_depth = [0.1, 0.5, 1.0, 2.0, 4.0]
    max_grad = np.max(np.abs(grads), axis=1)

    fig = go.Figure()
    fig.add_heatmap(x=well_depth, y=well_width, z=max_grad.reshape(len(well_width), len(well_depth)))
    fig.update_layout(xaxis_title='well_depth', yaxis_title='well_width')
    fig.show(renderer='browser')

    smoothest_run_ind = np.argmin(max_grad)
    dir = dirs[smoothest_run_ind]
    run_directory = head_directory.joinpath(Path(dir))
    log_file = run_directory.joinpath("screen.log")
    data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")
    data['E_pair_grad'] = np.gradient(data['E_pair'])
    data['P_grad'] = np.gradient(data['Press'])
    make_run_thermo_fig(data, dir, sigma)

    aa = 1

    '''
    try to predict local overlaps
    '''

    # energy = data['PotEng']
    #
    # num_lambda_steps = 100
    # #lambda_steps = optimize_lambda_spacing3(np.array(energy), num_lambda_steps=num_lambda_steps, buffer=50, maxiter=200)
    # lambda_steps = optimize_lambda_spacing(np.array(energy[::10]),
    #                                        num_lambda_steps=num_lambda_steps,
    #                                        buffer=10,
    #                                        maxiter=1000)


def make_run_thermo_fig(data, dir, sigma=0.01):
    from plotly.subplots import make_subplots
    rows = 2
    cols = 5
    keys_to_plot = ['E_pair', 'Volume', 'Temp', 'Press', 'KinEng', 'PotEng', 'TotEng', 'E_mol', 'E_pair_grad',
                    'P_grad']
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=keys_to_plot)
    for ind, k1 in enumerate(keys_to_plot):
        row = ind // cols + 1
        col = ind % cols + 1
        fig.add_scatter(x=data['Time'], y=gaussian_filter1d(data[k1], sigma), showlegend=False, row=row, col=col)
    fig.update_layout(title=dir)
    fig.show(renderer='browser')


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


def optimize_lambda_spacing(energy, num_lambda_steps, buffer, maxiter=200):
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

    run_directory = Path(r'D:\crystal_datasets\pscp\stage_three\generate_restart_files_lambda')
    log_file = run_directory.joinpath("screen.log")
    data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")

    lambdas = np.linspace(0, 1, len(energy))
    spacing_analysis_fig(data, energy, best_state, lambdas, buffer=buffer)

    fig = go.Figure(
        go.Scatter(x=np.linspace(0, 1, len(best_state)), y=best_state / best_state[-1], mode='lines+markers'))
    fig.add_scatter(x=np.linspace(0, 1, len(best_state)), y=np.linspace(0, 1, len(best_state))).show(renderer='browser')

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
