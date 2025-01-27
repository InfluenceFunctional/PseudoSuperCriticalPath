import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymbar
from openmm import unit
from plotly.subplots import make_subplots
from pymbar import timeseries
from tqdm import tqdm

from workflows.utils import compose_row_function


def free_energy(run_type: str,
                structure_name: str,
                reference_temperature: int,
                runs_directory: str,
                plot_trajs: bool = False,
                ):

    lmax = 1
    mbar_init = "BAR"  # "BAR" "zeros"

    structure_directory = Path(runs_directory).joinpath(structure_name)

    if not os.path.exists(structure_directory):
        assert False, "Structure directory does not exist - missing bulk runs"

    stage_directory = structure_directory.joinpath(run_type)
    if not os.path.exists(stage_directory):
        assert False, "Missing lambda generation trajectory directory!"

    lambda_gen_dir = stage_directory.joinpath(Path('gen_sampling'))
    lambda_restart_dir = stage_directory.joinpath(Path('restart_runs'))

    lambda_dirs = os.listdir(lambda_restart_dir)
    lambda_dirs = [dir for dir in lambda_dirs if "lambda_" in dir]
    lambda_steps = np.sort([float(dir.split('_')[-1]) for dir in lambda_dirs]).astype(int)

    temperature = float(reference_temperature)  # kelvin
    npt_tmp_path = structure_directory.joinpath(
        Path(f'npt_simulations/fluid/T_{reference_temperature}/tmp.out')
    )
    number_molecules = np.loadtxt(npt_tmp_path, skiprows=3, max_rows=1, dtype=int)[1]
    lambda_energies_path = stage_directory.joinpath(Path("lambda_energies.npy"))
    coul_series, gauss_series, lj_series, n_k, scale_series = load_lambda_energies(lambda_energies_path,
                                                                                   lambda_restart_dir, lambda_steps,
                                                                                   number_molecules)

    # convert from lambda steps to lambdas
    lambda_steps = scale_series[:, 3]
    lambdas = lambda_steps / lambda_steps[-1]
    lambdas = lambdas[lambdas <= lmax]

    # plot energies traj

    # compute total inter potential (with appropriate scaling) for full traj
    ukn = np.zeros((len(lambdas), len(coul_series)))
    beta = (1.0 / (unit.BOLTZMANN_CONSTANT_kB
                   * (temperature * unit.kelvin)
                   * unit.AVOGADRO_CONSTANT_NA)).value_in_unit(unit.kilocalorie_per_mole ** (-1))
    for i, lmbd in enumerate(lambdas):
        energy = (scale_series[i, 0] * coul_series +
                  scale_series[i, 1] * lj_series +
                  scale_series[i, 2] * gauss_series)

        ukn[i, :] = beta * energy

    max_ind = np.sum(n_k[:len(lambdas)])
    n_k = n_k[:len(lambdas)]
    ukn = ukn[:, :max_ind]

    k_vals = np.linspace(0, len(lambdas) - 2, 25).astype(int)
    batch = np.repeat(np.arange(len(lambdas)), n_k, axis=0)

    if plot_trajs:
        plot_effective_lambda_energy_traj(coul_series, gauss_series, lambdas, lj_series, n_k, number_molecules)
        plot_overlaps(batch, k_vals, ukn)
        plot_energy_traj(batch, lambdas, ukn)

    # free energy calculation
    max_samples = 100000
    sample_inds = []
    new_nk = []
    cum_nk = np.concatenate([np.zeros(1), np.cumsum(n_k)]).astype(int)
    for ind, n_of_k in enumerate(n_k):
        samples_to_add = np.linspace(0, n_of_k - 1, min(max_samples, n_of_k)).astype(int)
        sample_inds.append(cum_nk[ind] + samples_to_add)
        new_nk.append(len(samples_to_add))

    sample_inds = np.concatenate(sample_inds)
    new_ukn = ukn[:, sample_inds]
    new_nk = np.array(new_nk)

    mbar = pymbar.MBAR(new_ukn, new_nk, initialize=mbar_init, n_bootstraps=200)
    result = mbar.compute_free_energy_differences(uncertainty_method='bootstrap')

    # opt plot overlap matrix
    overlap = mbar.compute_overlap()
    nn_overlap = np.array([overlap['matrix'][i, i + 1] for i in range(len(overlap['matrix']) - 1)])
    fig = make_subplots(rows=1, cols=4)
    fig.add_scatter(x=lambdas, y=result['Delta_f'][0, :] / number_molecules,
                    error_y=dict(type='data', array=[result['dDelta_f'][0, :] / number_molecules], visible=True),
                    showlegend=False,
                    row=1, col=1
                    )

    fig.add_heatmap(z=overlap['matrix'], y=lambdas, x=lambdas, text=np.round(overlap['matrix'], 2),
                    texttemplate="%{text:.2g}", showscale=False, showlegend=False,
                    colorscale='blues', row=1, col=2)

    fig.add_scatter(x=lambdas, y=scale_series[:, 0], name='Coul Scale', row=1, col=3)
    fig.add_scatter(x=lambdas, y=scale_series[:, 1], name='LJ Scale', row=1, col=3)
    fig.add_scatter(x=lambdas, y=scale_series[:, 2], name='Gauss Scale', row=1, col=3)
    fig.add_scatter(x=lambdas[1:], y=nn_overlap, name="Nearest-Neighbor overlap", row=1, col=4)
    fig.update_layout(xaxis1_title='Lambda', yaxis1_title='Free Energy Difference',
                      xaxis2_title='Lambda', yaxis2_title='Lambda',
                      xaxis3_title='Lambda', yaxis3_title='Scaling Factor')
    fig.show(renderer='browser')


def plot_overlaps(batch, k_vals, ukn):
    fig = make_subplots(rows=5, cols=5, subplot_titles=[f"Lambda = {k / 100:.2g}" for k in k_vals])
    for ind in range(25):
        row = ind // 5 + 1
        col = ind % 5 + 1
        k_ind = k_vals[ind]

        fig.add_histogram(x=ukn[k_ind, batch == k_ind], nbinsx=50, row=row, col=col, name="Current",
                          legendgroup="Current", marker_color='red', showlegend=ind == 0)
        fig.add_histogram(x=ukn[k_ind + 1, batch == k_ind + 1], nbinsx=50, row=row, col=col, name="Next",
                          legendgroup="Next", marker_color='blue', showlegend=ind == 0)
    fig.update_xaxes(title='Energy')
    fig.show(renderer='browser')


def plot_energy_traj(batch, lambdas, ukn):
    means = np.zeros(len(lambdas))
    stds = np.zeros(len(lambdas))
    for ind in range(len(lambdas)):
        means[ind] = ukn[ind, batch == ind].mean()
        stds[ind] = ukn[ind, batch == ind].std()
    # plot lambda profile
    fig = go.Figure(
        go.Scatter(
            x=lambdas, y=means, mode='lines+markers', error_y=dict(type='data', array=stds ** 2, visible=True)
        )
    )
    fig.update_layout(xaxis_title='Lambda', yaxis_title='Energy')
    fig.show(renderer='browser')


def plot_effective_lambda_energy_traj(coul_series, gauss_series, lambdas, lj_series, n_k, number_molecules):
    # get mean energies of each type for each run
    c_ens = np.zeros(len(lambdas))
    place = 0
    for ind in range(len(lambdas)):
        c_ens[ind] = np.mean(coul_series[place:place + n_k[ind]])
        place += n_k[ind]
    lj_ens = np.zeros(len(lambdas))
    place = 0
    for ind in range(len(lambdas)):
        lj_ens[ind] = np.mean(lj_series[place:place + n_k[ind]])
        place += n_k[ind]
    g_ens = np.zeros(len(lambdas))
    place = 0
    for ind in range(len(lambdas)):
        g_ens[ind] = np.mean(gauss_series[place:place + n_k[ind]])
        place += n_k[ind]
    # plot effective lambda energy trajectory
    fig = go.Figure()
    fig.add_scatter(x=lambdas, y=c_ens / number_molecules, name='Coul Energy')
    fig.add_scatter(x=lambdas, y=lj_ens / number_molecules, name='LJ Energy')
    fig.add_scatter(x=lambdas, y=g_ens / number_molecules, name='Gauss Energy')
    fig.update_layout(title='Inter energies trajectory', xaxis_title='lambda', yaxis_title='Energy (kJ/mol)')
    fig.show(renderer='browser')


def load_lambda_energies(lambda_energies_path, lambda_restart_dir, lambda_steps, number_molecules):
    if os.path.isfile(lambda_energies_path):
        lambda_energies = np.load(lambda_energies_path, allow_pickle=True).item()
        coul_series = lambda_energies['coul']
        lj_series = lambda_energies['lj']
        gauss_series = lambda_energies['gauss']
        scale_series = lambda_energies['scales']
        n_k = lambda_energies['n_k']
        assert len(coul_series) == np.sum(n_k)
    else:
        print("Reading timeseries.")
        coul_series, gauss_series, lj_series, n_k, scale_series = (
            extract_run_inter_energies(lambda_restart_dir,
                                       lambda_steps,
                                       number_molecules))

        lambda_energies = {
            'coul': coul_series,
            'lj': lj_series,
            'gauss': gauss_series,
            'scales': scale_series,
            'n_k': n_k,
        }
        np.save(lambda_energies_path, lambda_energies)
        print("Finished reading timeseries.")
    return coul_series, gauss_series, lj_series, n_k, scale_series


def extract_run_inter_energies(lambda_restart_dir, lambda_steps, number_molecules):
    coul_series = np.empty(0)
    lj_series = np.empty(0)
    gauss_series = np.empty(0)
    n_k = np.empty(0, dtype=int)
    scale_series = np.zeros((len(lambda_steps), 4))
    for ind, lmbd in enumerate(tqdm(lambda_steps)):
        run_directory = lambda_restart_dir.joinpath(Path(f"lambda_{lmbd}"))
        assert (np.loadtxt(run_directory.joinpath("tmp.out"), skiprows=3, max_rows=1, dtype=int)[1]
                == number_molecules)
        log_file = run_directory.joinpath("screen.log")
        # noinspection PyTypeChecker
        try:
            data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")
        except ValueError as e:
            if str(e) == "No valid rows found":
                lambda_steps = lambda_steps[lambda_steps != lmbd]
                print(str(e))
                continue
            else:
                raise e

        coul_energy = (data['c_coul'] + data['E_long'] / data['v_scale_coulomb']).to_numpy() * unit.kilocalorie_per_mole
        gauss_energy = data['c_gauss'].to_numpy() * unit.kilocalorie_per_mole
        lj_energy = data['c_lj'].to_numpy() * unit.kilocalorie_per_mole
        scale_series[ind, :] = [data['v_scale_coulomb'][0], data['v_scale_lj'][0], data['v_scale_gauss'][0], lmbd]

        data_full = (coul_energy + gauss_energy + lj_energy)

        t0, g, Neff_max = timeseries.detect_equilibration(data_full)  # compute indices of uncorrelated timeseries

        coul_indices = timeseries.subsample_correlated_data(coul_energy[t0:], g=g)
        coul_data = coul_energy[t0:][coul_indices]

        lj_indices = timeseries.subsample_correlated_data(lj_energy[t0:], g=g)
        lj_data = lj_energy[t0:][lj_indices]

        gauss_indices = timeseries.subsample_correlated_data(gauss_energy[t0:], g=g)
        gauss_data = gauss_energy[t0:][gauss_indices]

        n_k = np.append(n_k, len(gauss_data))
        coul_series = np.append(coul_series, coul_data)
        lj_series = np.append(lj_series, lj_data)
        gauss_series = np.append(gauss_series, gauss_data)

    scale_series = np.delete(scale_series, np.argwhere(scale_series.sum(axis=1) == 0), axis=0)  # delete unused rows
    assert len(coul_series) == np.sum(n_k)

    return coul_series, gauss_series, lj_series, n_k, scale_series
