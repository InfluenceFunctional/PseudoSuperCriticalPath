import os
from copy import copy
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymbar
from openmm import unit
from plotly.subplots import make_subplots
from pymbar import timeseries
from scipy.ndimage import gaussian_filter1d

from analyze import compose_row_function
from pathlib import Path
from tqdm import tqdm

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

    import plotly.graph_objects as go
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


# for lmbd in [0.68, 0.69, 0.70, 0.71, 0.72]:
#     plot_run_thermo(lmbd)


def unit_piecewise_linear(num_points, points, num_gridpoints):
    #num_gridpoints = 100
    xvals = np.linspace(0.001, 0.999, num_gridpoints)

    #num_points = 10
    #points = np.linspace(0.1, 0.9, num_points) + np.random.randn(num_points) / 10
    all_points = np.concatenate([np.zeros(1), points, np.ones(1)])

    start_point = 1 / (num_points + 1)
    xs = np.linspace(start_point, 1 - start_point, num_points)
    xs = np.concatenate([np.zeros(1), xs, np.ones(1)])

    slopes = np.zeros(len(xs) - 1)
    intercepts = np.zeros_like(slopes)
    for ind in range(1, len(xs)):
        slopes[ind - 1] = (all_points[ind] - all_points[ind - 1]) / (xs[ind] - xs[ind - 1])
        intercepts[ind - 1] = all_points[ind] - slopes[ind - 1] * xs[ind]

    conditions = [(xvals > xs[ind]) * (xvals < xs[ind + 1]) for ind in range(len(xs) - 1)]
    from copy import deepcopy
    linear = lambda m, x, b: m * x + b
    functions = [lambda x, m=slopes[ind], b=intercepts[ind]: linear(m, x, b) for ind in range(len(slopes))]

    pf = np.piecewise(xvals, conditions, functions)

    # test fit
    # import plotly.graph_objects as go
    # fig = go.Figure(go.Scatter(x=xvals, y=pf))
    # fig.add_scatter(x=xs, y=all_points)
    # fig.show()
    return pf


def unit_piecewise_linear(num_points, points, num_gridpoints):
    #num_gridpoints = 100
    xvals = np.linspace(0.001, 0.999, num_gridpoints)

    #num_points = 10
    #points = np.linspace(0.1, 0.9, num_points) + np.random.randn(num_points) / 10
    all_points = np.concatenate([np.zeros(1), points, np.ones(1)])

    start_point = 1 / (num_points + 1)
    xs = np.linspace(start_point, 1 - start_point, num_points)
    xs = np.concatenate([np.zeros(1), xs, np.ones(1)])

    slopes = np.zeros(len(xs) - 1)
    intercepts = np.zeros_like(slopes)
    for ind in range(1, len(xs)):
        slopes[ind - 1] = (all_points[ind] - all_points[ind - 1]) / (xs[ind] - xs[ind - 1])
        intercepts[ind - 1] = all_points[ind] - slopes[ind - 1] * xs[ind]

    conditions = [(xvals > xs[ind]) * (xvals < xs[ind + 1]) for ind in range(len(xs) - 1)]
    from copy import deepcopy
    linear = lambda m, x, b: m * x + b
    functions = [lambda x, m=slopes[ind], b=intercepts[ind]: linear(m, x, b) for ind in range(len(slopes))]

    pf = np.piecewise(xvals, conditions, functions)

    # test fit
    # import plotly.graph_objects as go
    # fig = go.Figure(go.Scatter(x=xvals, y=pf))
    # fig.add_scatter(x=xs, y=all_points)
    # fig.show()
    return pf


def analyze_spacing(energy, num_lambdas, spline_points):
    lambda_steps = (unit_piecewise_linear(len(spline_points), spline_points, num_lambdas) * (len(energy) - 1)).astype(
        int)
    lambda_energies, lambda_widths, nn_overlaps, overlaps = (
        old_compute_lambda_overlaps(energy, lambda_steps, num_lambdas))
    return lambda_steps, nn_overlaps


def opt_analyze_spacing(spline_points, energy):
    steps, overlaps = analyze_spacing(energy, 100, spline_points)
    return -np.amin(gaussian_filter1d(overlaps, sigma=1))


def sigmoid(x, scale: float = 2):
    return (np.tanh((x - 0.5) * scale) + 1) / 2


def mini_mc(energy: np.ndarray, num_lambdas: int, steps: int = 1000, dim: int = 10, T: float = 1,
            step_size: float = 0.01):
    x_rec = []
    overlaps_rec = []
    score_rec = []
    x0 = np.ones(dim) * (1 / (dim + 1))
    prev_score = 100
    for _ in tqdm(range(steps)):
        x_prop = (x0 + np.random.randn(dim) * step_size).clip(min=0.01)
        x_sub = sigmoid(np.cumsum(x_prop))  # force on to 0-1
        _, overlaps = analyze_spacing(energy, num_lambdas, spline_points=x_sub)
        #score = -np.amin(gaussian_filter1d(overlaps, sigma=1))
        score = np.var(overlaps)
        delta = score - prev_score
        rand = np.random.rand(1)
        acceptance_ratio = min(1, np.nan_to_num(np.exp(-delta / T)))
        if acceptance_ratio > rand:
            x0 = x_prop
            x_rec.append(x_sub)
            overlaps_rec.append(overlaps)
            score_rec.append(score)
            prev_score = copy(score)
        else:
            pass

    x_record = np.vstack(x_rec)
    overlaps_record = np.vstack(overlaps_rec)
    score_record = np.array(score_rec)

    best_ind = np.argmin(score_record)
    best_sample = x_record[best_ind]

    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(y=score_rec)).show()
    fig = go.Figure(go.Scatter(y=overlaps_record[best_ind])).show()

    return best_sample, x_record, overlaps_record


def analyze_gen_runs():
    run_directory = Path(r'D:\crystal_datasets\pscp\stage_three\generate_restart_files_lambda')
    log_file = run_directory.joinpath("screen.log")
    data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")

    import plotly.graph_objects as go
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
    lambdas = np.linspace(0, 1, len(energy))

    #best_steps, x_record, overlap_record = mini_mc(energy, num_lambdas, 100, dim=10, T=0.0001, step_size=0.05)
    #lambda_steps, overlaps = analyze_spacing(energy, num_lambdas, best_steps)
    #fig = go.Figure(go.Scatter(x=lambdas[lambda_steps], y=best_steps, mode='markers')).show()
    num_lambda_steps = 100
    #lambda_steps = optimize_lambda_spacing3(np.array(energy), num_lambda_steps=num_lambda_steps, buffer=50, maxiter=200)
    lambda_steps = optimize_lambda_spacing3(np.array(energy[::10]), num_lambda_steps=num_lambda_steps, buffer=5, maxiter=200)

    #spacing_analysis_fig(data, energy, lambda_steps, lambdas, num_lambda_steps)

    aa = 1


def spacing_analysis_fig(data, energy, lambda_steps, lambdas, num_lambda_steps):
    lambda_energies, lambda_widths, nn_overlaps, overlaps = (
        old_compute_lambda_overlaps(energy, lambda_steps, num_lambda_steps))
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


def old_compute_lambda_overlaps(energy, lambda_steps, num_lambdas):
    lambda_energies = np.zeros(num_lambdas)
    lambda_widths = np.zeros(num_lambdas)
    for ind in range(num_lambdas):
        i1 = max(0, ind - 1)
        i2 = min(num_lambdas - 1, ind + 1)
        s1 = lambda_steps[i1]
        s2 = lambda_steps[ind]
        s3 = lambda_steps[i2]

        r0 = s1 + (s2 - s1) // 2
        r1 = s3 - (s3 - s2) // 2

        step_inds = np.arange(
            r0, r1
        )

        lambda_energies[ind] = np.mean(energy[step_inds])
        lambda_widths[ind] = np.std(energy[step_inds])
    overlaps = np.zeros((num_lambdas, num_lambdas))
    for ind1 in range(num_lambdas):
        for ind2 in range(num_lambdas):
            e1 = lambda_energies[ind1]
            e2 = lambda_energies[ind2]
            s1 = lambda_widths[ind1]
            s2 = lambda_widths[ind2]
            try:
                overlaps[ind1, ind2] = NormalDist(mu=e1, sigma=s1).overlap(NormalDist(mu=e2, sigma=s2))
            except:
                pass
    nn_overlaps = np.array(
        [overlaps[ind, ind - 1] for ind in range(1, num_lambdas)]
    )

    return lambda_energies, lambda_widths, nn_overlaps, overlaps


def compute_lambda_overlaps(energy, energy_spacing):
    steps_to_sample = np.arange(len(energy))[::energy_spacing]
    buffer = 50
    lambda_energies = np.zeros(len(steps_to_sample))
    lambda_widths = np.zeros(len(steps_to_sample))
    lambda_steps = steps_to_sample
    for ind in range(len(steps_to_sample)):
        i1 = max(0, ind - buffer)
        i2 = min(len(steps_to_sample) - 1, ind + buffer)
        s1 = lambda_steps[i1]
        s2 = lambda_steps[ind]
        s3 = lambda_steps[i2]

        r0 = s1 + (s2 - s1) // 2
        r1 = s3 - (s3 - s2) // 2

        step_inds = np.arange(
            r0, r1
        )

        lambda_energies[ind] = np.mean(energy[step_inds])
        lambda_widths[ind] = np.std(energy[step_inds])

    overlaps = np.exp(
        -(lambda_energies[:, None] - lambda_energies[None, :]) ** 2 / (lambda_widths[:, None] + lambda_widths[None, :]))

    nn_overlaps = np.array(
        [overlaps[ind, ind - 1] for ind in range(1, len(steps_to_sample))]
    )
    return lambda_energies, lambda_widths, nn_overlaps, overlaps


def optimize_lambda_spacing(energy, num_lambda_steps, num_optim_steps):
    steps_to_sample = np.linspace(0, len(energy) - 1, num_lambda_steps).astype(int)
    buffer = 1
    tolerance = 0.8

    overlaps_record = []
    steps_record = []
    scores_record = []
    delta_step = 2
    for iter_ind in tqdm(range(num_optim_steps)):
        lambda_energies, lambda_widths, nn_overlaps, overlaps = batch_compute_overlaps(buffer, energy, steps_to_sample)
        iter_ind += 1
        mean_overlap = nn_overlaps.mean()
        overlaps_record.append(nn_overlaps)
        steps_record.append(steps_to_sample * 1)
        scores_record.append(nn_overlaps.min())
        rands_to_check = np.random.choice(np.arange(1, num_lambda_steps), num_lambda_steps // 4, replace=False)
        for ind in rands_to_check:
            ov = nn_overlaps[ind - 1]
            # if too big, separate the points,
            # if too small, bring the points together
            i0 = ind - 1
            i1 = ind
            delta = steps_to_sample[i1] - steps_to_sample[i0]
            if ov < mean_overlap * tolerance and delta > delta_step + 1:
                # maintain fixed endpoints
                if i0 != 0:
                    steps_to_sample[i0] += 1
                if i1 != num_lambda_steps - 1:
                    steps_to_sample[i1] -= 1
            elif ov > mean_overlap / tolerance:
                if i0 != 0:
                    delta2 = steps_to_sample[i0] - steps_to_sample[ind - 2]
                    if delta2 > delta_step + 1:
                        steps_to_sample[i0] -= 1
                if i1 != num_lambda_steps - 1:
                    delta2 = steps_to_sample[ind + 1] - steps_to_sample[ind]
                    if delta2 > delta_step + 1:
                        steps_to_sample[i1] += 1

        if iter_ind % 10 == 0 and iter_ind > 100:
            converged = np.var(np.vstack(steps_record)[-10:], axis=0).max() < 1e-2
            if converged:
                break

    overlaps_record = np.vstack(overlaps_record)
    steps_record = np.vstack(steps_record)
    best_sample = steps_record[np.argmax(scores_record)]

    import plotly.graph_objects as go
    go.Figure(go.Heatmap(z=overlaps)).show(renderer='browser')
    go.Figure(go.Scatter(x=best_sample, y=overlaps_record[np.argmax(scores_record)], name='nn overlaps',
                         showlegend=True)).show(renderer='browser')
    go.Figure(go.Scatter(y=overlaps_record.min(1), name='min overlap', showlegend=True)).show(renderer='browser')
    go.Figure(go.Scatter(y=overlaps_record.mean(1), name='mean overlap', showlegend=True)).show(
        renderer='browser')
    fig = go.Figure()
    for ind in range(steps_record.shape[1]):
        fig.add_scatter(y=steps_record[:, ind] - steps_record[0, ind])
    fig.show(renderer='browser')
    go.Figure(go.Scatter(y=steps_record[-1] - steps_record[0])).show(renderer='browser')
    return best_sample


def optimize_lambda_spacing2(energy, num_lambda_steps):
    buffer = 50

    def overlap_func(steps_in, buffer=buffer, energy=energy):
        lambdas = sigmoid(np.cumsum(sigmoid(steps_in, scale=4)), scale=4)
        lambdas = np.concatenate([
            np.zeros(1),
            lambdas,
            np.ones(1),
        ])
        steps_to_sample = (lambdas * (len(energy) - 1)).astype(int)

        lambda_energies, lambda_widths, nn_overlaps, overlaps = batch_compute_overlaps(buffer, energy, steps_to_sample)
        return nn_overlaps.var()

    num_lambda_steps = 5
    x0 = np.ones(num_lambda_steps - 2) / num_lambda_steps
    res = optimize.basinhopping(
        overlap_func, x0, minimizer_kwargs={'args': [buffer, energy]},
    )
    best_sample = res.x

    lambdas = sigmoid(np.cumsum(sigmoid(best_sample, scale=4)), scale=4)
    lambdas = np.concatenate([
        np.zeros(1),
        lambdas,
        np.ones(1),
    ])
    steps_to_sample = (lambdas * (len(energy) - 1)).astype(int)

    #lambda_energies, lambda_widths, nn_overlaps, overlaps = batch_compute_overlaps(buffer, energy, steps_to_sample)

    # import plotly.graph_objects as go
    # fig = go.Figure(go.Heatmap(z=(overlaps))).show(renderer='browser')
    # fig = go.Figure(go.Scatter(x=steps_to_sample, y=nn_overlaps, name='nn overlaps', showlegend=True)).show(renderer='browser')

    return steps_to_sample


def optimize_lambda_spacing3(energy, num_lambda_steps, buffer, maxiter=200):
    all_steps = np.arange(0, len(energy))
    lambda_energies, lambda_widths, full_nn_overlaps, full_overlaps = batch_compute_overlaps(buffer, energy, all_steps)

    naive_steps = np.linspace(0, len(energy) - 1, num_lambda_steps).astype(int)

    iter = 0
    tolerance = 1
    converged = False
    overlaps_record = []
    states_record = []
    while not converged and iter < maxiter:
        if iter == 0:
            current_steps = np.copy(naive_steps)

        lambda_energies, lambda_widths, nn_overlaps, overlaps = (
            batch_compute_overlaps(buffer, energy, current_steps))

        if nn_overlaps.min() < nn_overlaps.mean() * 0.8:
            tolerance *= 0.975

        target_overlap = nn_overlaps.mean() * tolerance
        overlaps_record.append(nn_overlaps)
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

    mins = []
    opt_exponents = np.linspace(0.1, 1.0, 40)
    for exp in opt_exponents:
        best_state = np.asarray(states_record[np.argmax(min_record)])
        best_state = ((best_state / len(energy)) ** exp) * len(energy)  # rescale slightly
        best_state = best_state.astype(int)
        lambda_energies, lambda_widths, nn_overlaps, overlaps = (
            batch_compute_overlaps(buffer, energy, best_state))
        mins.append(nn_overlaps.min())

    exp = opt_exponents[np.argmax(np.array(mins))]
    best_state = np.asarray(states_record[np.argmax(min_record)])
    best_state = ((best_state / len(energy)) ** exp) * len(energy)  # rescale slightly
    best_state = best_state.astype(int)

    run_directory = Path(r'D:\crystal_datasets\pscp\stage_three\generate_restart_files_lambda')
    log_file = run_directory.joinpath("screen.log")
    data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")

    lambdas = np.linspace(0, 1, len(energy))
    spacing_analysis_fig(data, energy, best_state, lambdas, len(best_state))

    return best_state


def batch_compute_overlaps(buffer, energy, steps_to_sample):
    lambda_energies = np.zeros(len(steps_to_sample))
    lambda_widths = np.zeros(len(steps_to_sample))
    lambda_steps = steps_to_sample
    for ind in range(len(steps_to_sample)):
        s1 = max(lambda_steps[ind] - buffer, 0)
        s2 = lambda_steps[ind]
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
