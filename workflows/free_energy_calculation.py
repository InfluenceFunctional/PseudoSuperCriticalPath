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

    temperature = float(reference_temperature)  # kelvin
    npt_tmp_path = structure_directory.joinpath(
        Path(f'npt_simulations/fluid/T_{reference_temperature}/tmp.out')
    )

    number_molecules = np.loadtxt(npt_tmp_path, skiprows=3, max_rows=1, dtype=int)[1]
    lambda_energies_path = stage_directory.joinpath(Path("lambda_energies.npy"))

    if run_type == 'stage_two':
        lambda_dirs = os.listdir(lambda_restart_dir)
        lambda_dirs = [dir for dir in lambda_dirs if "lambda_" in dir]
        lambdas = np.sort([float(dir.split('_')[-1].replace('-', '.')) for dir in lambda_dirs])
        lambdas = lambdas[lambdas <= lmax]

        coul_series, gauss_series, lj_series, n_k, scale_series = load_lambda_energies(lambda_energies_path,
                                                                                       lambda_restart_dir, lambdas,
                                                                                       number_molecules)

        mbar_init = 'zeros'
        result = volumetric_free_energy_calculation(lambda_restart_dir, lambdas, number_molecules,
                                                    reference_temperature,
                                                    stage_directory, mbar_init, plot_trajs, coul_series, gauss_series,
                                                    lj_series,
                                                    n_k)

    else:
        # convert from lambda steps to lambdas
        lambda_dirs = os.listdir(lambda_restart_dir)
        lambda_dirs = [dir for dir in lambda_dirs if "lambda_" in dir]

        lambda_steps = np.sort([float(dir.split('_')[-1]) for dir in lambda_dirs]).astype(int)

        lambdas = lambda_steps / lambda_steps[-1]
        lambdas = lambdas[lambdas <= lmax]

        coul_series, gauss_series, lj_series, n_k, scale_series = load_lambda_energies(lambda_energies_path,
                                                                                       lambda_restart_dir, lambda_steps,
                                                                                       number_molecules)

        result = alchemical_free_energy_calculation(coul_series, gauss_series, lambdas, lj_series, mbar_init, n_k,
                                                    number_molecules, plot_trajs, scale_series, temperature)

    np.save(structure_directory.joinpath(run_type + '_free_energy'), result)


def volumetric_free_energy_calculation(lambda_restart_dir, lambdas, number_molecules, reference_temperature,
                                       stage_directory, mbar_init, plot_trajs, coul_series, gauss_series, lj_series,
                                       old_n_k):
    box_energies_path = stage_directory.joinpath(Path("box_energies.npy"))

    run_directory = lambda_restart_dir.joinpath(Path(f"lambda_{1}"))
    restart_files = os.listdir(run_directory)
    restart_files = [dir for dir in restart_files if "stage_two_lambda_sample.restart" in dir]
    num_restarts = len(restart_files)
    if not os.path.exists(box_energies_path):

        # process re_boxed energies
        all_box_potentials = np.zeros((len(lambdas), num_restarts, len(lambdas)))
        all_box_volumes = np.zeros((len(lambdas), num_restarts, len(lambdas)))
        for ind, lmbd in enumerate(tqdm(lambdas)):
            lmbd_str = f"{lmbd:.3g}".replace('.', '-')
            run_directory = lambda_restart_dir.joinpath(Path(f"lambda_{lmbd_str}"))
            assert (np.loadtxt(run_directory.joinpath("tmp.out"), skiprows=3, max_rows=1, dtype=int)[1]
                    == number_molecules)

            snapshot_files = os.listdir(run_directory)
            snapshot_files = [file for file in snapshot_files if ('.log' in file and 'screen' not in file)]

            for f_ind, file in enumerate(snapshot_files):
                ff = file[:-4].split('_')
                i1, i2, i3 = int(ff[0]), int(ff[1]), int(ff[2])
                log_file = run_directory.joinpath(file)
                data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")
                all_box_potentials[i1, i2, i3] = data['PotEng'][0]
                all_box_volumes[i1, i2, i3] = data['Volume'][0]
        N = number_molecules
        beta = (1.0 / (unit.BOLTZMANN_CONSTANT_kB
                       * (reference_temperature * unit.kelvin)
                       * unit.AVOGADRO_CONSTANT_NA)).value_in_unit(unit.kilocalorie_per_mole ** (-1))
        reduced_pot = -N * np.log(all_box_volumes) + all_box_potentials * beta
        np.save(box_energies_path, reduced_pot)
    else:
        reduced_pot = np.load(box_energies_path, allow_pickle=True)

    '''free energy calculation'''
    ukn = np.zeros((len(lambdas), reduced_pot.shape[-1] * reduced_pot.shape[-2]))
    for ind in range(len(lambdas)):
        ukn[ind] = reduced_pot[ind].T.flatten()

    n_k = np.ones(len(lambdas)).astype(int) * num_restarts

    k_vals = np.linspace(0, len(lambdas) - 2, 25).astype(int)
    batch = np.repeat(np.arange(len(lambdas)), n_k, axis=0)
    flat_pot = []
    for iL in range(len(lambdas)):
        flat_pot.extend(reduced_pot[iL, :, iL])
    flat_pot = np.array(flat_pot)

    if plot_trajs:
        plot_effective_lambda_energy_traj(coul_series, gauss_series, lambdas, lj_series, old_n_k, number_molecules)
        plot_effective_lambda_energy_traj(flat_pot, flat_pot, lambdas, flat_pot, n_k, number_molecules)

        plot_overlaps(batch, k_vals, ukn, num_lambdas=len(lambdas))
        plot_energy_traj(batch, lambdas, ukn)

    mbar = pymbar.MBAR(ukn / number_molecules, n_k, initialize=mbar_init)  #, n_bootstraps=50)
    result = mbar.compute_free_energy_differences()  #uncertainty_method='bootstrap')
    if plot_trajs:
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
        fig.add_scatter(x=lambdas, y=flat_pot, name='Reduced Pot', row=1, col=3)

        fig.add_scatter(x=lambdas[1:], y=nn_overlap, name="Nearest-Neighbor overlap", row=1, col=4)
        fig.update_layout(xaxis1_title='Lambda', yaxis1_title='Free Energy Difference',
                          xaxis2_title='Lambda', yaxis2_title='Lambda',
                          xaxis3_title='Lambda', yaxis3_title='Scaling Factor')
        fig.show(renderer='browser')

    return result


def alchemical_free_energy_calculation(coul_series, gauss_series, lambdas, lj_series, mbar_init, n_k, number_molecules,
                                       plot_trajs, scale_series, temperature):
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
    mbar = pymbar.MBAR(new_ukn, new_nk, initialize=mbar_init)  #, n_bootstraps=200)
    result = mbar.compute_free_energy_differences()  #uncertainty_method='bootstrap')
    if plot_trajs:
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
    return result


def plot_overlaps(batch, k_vals, ukn, num_lambdas=100):
    fig = make_subplots(rows=5, cols=5, subplot_titles=[f"Lambda = {k / num_lambdas:.2g}" for k in k_vals])
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
            x=lambdas, y=means, mode='lines+markers', error_y=dict(type='data', array=stds, visible=True)
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


def load_lambda_energies(lambda_energies_path, lambda_restart_dir, lambdas, number_molecules):
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
                                       lambdas,
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


def extract_run_inter_energies(lambda_restart_dir, lambdas, number_molecules):
    coul_series = np.empty(0)
    lj_series = np.empty(0)
    gauss_series = np.empty(0)
    n_k = np.empty(0, dtype=int)
    scale_series = np.zeros((len(lambdas), 4))
    for ind, lmbd in enumerate(tqdm(lambdas)):
        lmbd_str = f"{lmbd:.3g}".replace('.', '-')
        run_directory = lambda_restart_dir.joinpath(Path(f"lambda_{lmbd_str}"))
        assert (np.loadtxt(run_directory.joinpath("tmp.out"), skiprows=3, max_rows=1, dtype=int)[1]
                == number_molecules)
        log_file = run_directory.joinpath("screen.log")
        # noinspection PyTypeChecker
        try:
            data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, False), sep=r"\s+")
        except ValueError as e:
            if str(e) == "No valid rows found":
                lambdas = lambdas[lambdas != lmbd]
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


def relative_free_energy(
        structure_name: str,
        reference_temperature: int,
        runs_directory: str,
        temperature_span: int,
        temperature_delta: int,
        plot_trajs: bool,
):
    minimum_temperature = reference_temperature - temperature_span  # Kelvin
    maximum_temperature = reference_temperature + temperature_span  # Kelvin
    pressure = 1.0  # atm
    temperatures = np.arange(minimum_temperature, maximum_temperature + temperature_delta, temperature_delta)

    structure_directory = Path(runs_directory).joinpath(structure_name)

    if not os.path.exists(structure_directory):
        assert False, "Structure directory does not exist - missing bulk runs"

    fluid_directory = structure_directory.joinpath("npt_simulations/fluid")
    solid_directory = structure_directory.joinpath("npt_simulations/solid")

    npt_tmp_path = structure_directory.joinpath(
        Path(f'npt_simulations/fluid/T_{reference_temperature}/tmp.out')
    )
    number_molecules = np.loadtxt(npt_tmp_path, skiprows=3, max_rows=1, dtype=int)[1]
    for directory_index, (directory, state) in enumerate(zip((fluid_directory, solid_directory), ('fluid', 'solid'))):
        energies_path = structure_directory.joinpath(f"{state}_bulk_energies.npy")

        if os.path.isfile(energies_path):
            pass

        else:
            print("Reading timeseries.")
            data_series = np.empty(0)
            n_k = np.empty(0, dtype=int)
            for temperature in tqdm(temperatures):
                run_dir = directory.joinpath(f"T_{temperature}")
                tmp_path = run_dir.joinpath('tmp.out')
                screen_path = run_dir.joinpath('screen.log')
                assert (np.loadtxt(tmp_path, skiprows=3, max_rows=1, dtype=int)[1] == number_molecules)
                log_file = screen_path
                # noinspection PyTypeChecker
                data = pd.read_csv(log_file, skiprows=compose_row_function(log_file, True), sep=r"\s+")
                data_full = (data["PotEng"].to_numpy() * unit.kilocalorie_per_mole +
                             pressure * data["Volume"].to_numpy()
                             * unit.AVOGADRO_CONSTANT_NA * (unit.atmosphere * unit.angstrom ** 3))
                t0, g, Neff_max = timeseries.detect_equilibration(
                    data_full)  # compute indices of uncorrelated timeseries
                data_equil = data_full[t0:]
                indices = timeseries.subsample_correlated_data(data_equil, g=g)
                data = data_equil[indices]
                n_k = np.append(n_k, len(data))
                data_series = np.append(data_series, data)
            assert len(data_series) == np.sum(n_k)

        state_energies = {'PotEng': data_series, 'n_k': n_k}
        np.save(energies_path, state_energies)
        print("Finished reading timeseries.")

    # eqn 2 in the paper

    results_list = []
    overlaps_list = []
    for directory_index, (directory, state) in enumerate(zip((fluid_directory, solid_directory), ('fluid', 'solid'))):
        energies_path = structure_directory.joinpath(f"{state}_bulk_energies.npy")
        state_energies = np.load(energies_path, allow_pickle=True).item()
        data_series = state_energies['PotEng']
        n_k = state_energies['n_k']
        assert len(data_series) == np.sum(n_k)

        ukn = np.zeros((len(temperatures), len(data_series)))
        for i, temperature in enumerate(temperatures):
            beta = (1.0 / (unit.BOLTZMANN_CONSTANT_kB
                           * (temperature * unit.kelvin)
                           * unit.AVOGADRO_CONSTANT_NA)).value_in_unit(unit.kilocalorie_per_mole ** (-1))
            ukn[i, :] = beta * data_series[:]

        mbar = pymbar.MBAR(ukn, n_k, initialize="BAR")
        overlap = mbar.compute_overlap()
        result = mbar.compute_free_energy_differences()
        results_list.append(result)
        overlaps_list.append(overlap)

    if plot_trajs:
        fig = make_subplots(rows=1, cols=4)
        for phase_ind, (result, overlap, state) in enumerate(zip(results_list, overlaps_list, ['fluid', 'solid'])):
            nn_overlap = np.array([overlap['matrix'][i, i + 1] for i in range(len(overlap['matrix']) - 1)])

            if state == 'fluid':
                fig.add_scatter(x=temperatures, y=result['Delta_f'][0, :] / number_molecules,
                                error_y=dict(type='data', array=[result['dDelta_f'][0, :] / number_molecules],
                                             visible=True),
                                showlegend=True,
                                name=state, legendgroup=state, marker_color='blue',
                                row=1, col=1
                                )
                fig.add_heatmap(z=overlap['matrix'], y=temperatures, x=temperatures,
                                text=np.round(overlap['matrix'], 2),
                                texttemplate="%{text:.2g}", showscale=False, showlegend=False,
                                colorscale='blues', row=1, col=2)

                fig.add_scatter(x=temperatures[1:], y=nn_overlap, showlegend=False, marker_color='blue',
                                name=state, legendgroup=state, row=1, col=4)
            else:
                fig.add_scatter(x=temperatures, y=result['Delta_f'][0, :] / number_molecules,
                                error_y=dict(type='data', array=[result['dDelta_f'][0, :] / number_molecules],
                                             visible=True),
                                showlegend=True,
                                name=state, legendgroup=state, marker_color='red',
                                row=1, col=1
                                )
                fig.add_heatmap(z=overlap['matrix'], y=temperatures, x=temperatures,
                                text=np.round(overlap['matrix'], 2),
                                texttemplate="%{text:.2g}", showscale=False, showlegend=False,
                                colorscale='blues', row=1, col=3)

                fig.add_scatter(x=temperatures[1:], y=nn_overlap, showlegend=False, marker_color='red',
                                name=state, legendgroup=state, row=1, col=4)
            fig.update_layout(xaxis1_title='Temperature (K)', yaxis1_title='Free Energy Difference',
                              xaxis2_title='Temperature (K)', yaxis2_title='Temperature (K)',
                              xaxis3_title='Temperature (K)', yaxis3_title='Temperature (K)',
                              xaxis4_title='Temperature (K)', yaxis4_title='Nearest-Neighbor Overlap')

        fig.show(renderer='browser')
    # plt.errorbar(temperatures, result["Delta_f"][0, :] / number_molecules,
    #              yerr=result["dDelta_f"][0, :] / number_molecules, label=directory,
    #              color=f"C{directory_index}")
    # entropy = -np.gradient(result["Delta_f"][0, :], temperature_step)
    # heat_capacity = temperatures * np.gradient(entropy, temperature_step)
    # plt.plot(temperatures, -np.gradient(result["Delta_f"][0, :], temperature_step),
    #          color=f"C{directory_index}", linestyle="--")
    # plt.plot(temperatures, heat_capacity, color=f"C{directory_index}", linestyle=":")

    np.save(structure_directory.joinpath('fluid_free_energy'), results_list[0])
    np.save(structure_directory.joinpath('solid_free_energy'), results_list[1])


def loop_free_energy(
        structure_name: str,
        reference_temperature: int,
        runs_directory: str,
        temperature_span: int,
        temperature_delta: int
):
    structure_directory = Path(runs_directory).joinpath(structure_name)

    npt_tmp_path = structure_directory.joinpath(
        Path(f'npt_simulations/fluid/T_{reference_temperature}/tmp.out')
    )
    number_molecules = np.loadtxt(npt_tmp_path, skiprows=3, max_rows=1, dtype=int)[1]

    minimum_temperature = reference_temperature - temperature_span  # Kelvin
    maximum_temperature = reference_temperature + temperature_span  # Kelvin
    temperatures = np.arange(minimum_temperature, maximum_temperature + temperature_delta, temperature_delta)

    dgs = []
    ddgs = []
    for stage in ['stage_one', 'stage_two', 'stage_three', 'fluid', 'solid']:
        results_path = structure_directory.joinpath(stage + '_free_energy.npy')
        results = np.load(results_path, allow_pickle=True).item()
        dG = results['Delta_f']
        ddG = results['dDelta_f']
        if stage != 'stage_two':
            dG /= number_molecules
            ddG /= number_molecules
        dgs.append(dG[0, :])
        ddgs.append(ddG[0, :])

    fig = go.Figure()
    ylevel = 0
    xlevel = 0
    stage_names = ['stage one', 'stage two', 'stage three']
    for ind in range(3):
        fig.add_scatter(x=np.arange(len(dgs[ind])) + xlevel,
                        y=dgs[ind] - dgs[ind][0] + ylevel,
                        name=stage_names[ind], )
        ylevel += dgs[ind][-1]
        xlevel += len(dgs[ind]) - 1
    fig.add_scatter(x=[0, xlevel], y=[0, ylevel], mode='lines', name='Overall')
    fig.update_layout(title='Alchemical Profile', yaxis_title='Free Energy', xaxis_title="Lambda Steps")
    fig.show(renderer='browser')

    dG_T_ref = -ylevel
    dF_bulk = dgs[-1] - dgs[-2]
    tref_ind = np.argwhere(temperatures == reference_temperature).flatten()
    dF_ref = dF_bulk[tref_ind]
    # from equation 4, dG(l,s) = kbT[dF(T)-dF(T_ref)] + (T/T_ref)*dG(T_ref)
    temp_free_energies = np.zeros(len(temperatures))
    bulk_ddF = np.zeros(len(temperatures))
    dG_T = np.zeros(len(temperatures))
    for ind, temp in enumerate(temperatures):
        beta = (1.0 / (unit.BOLTZMANN_CONSTANT_kB
                       * (temp * unit.kelvin)
                       * unit.AVOGADRO_CONSTANT_NA)).value_in_unit(unit.kilocalorie_per_mole ** (-1))
        bulk_ddF[ind] = (dF_bulk[ind] - dF_ref) / beta
        dG_T[ind] = (temp / reference_temperature) * dG_T_ref
        temp_free_energies[ind] = (dF_bulk[ind] - dF_ref) / beta + (temp / reference_temperature) * dG_T_ref

    interp_temps = np.linspace(temperatures[0], temperatures[-1], 10000)
    dg_interps = np.interp(interp_temps, temperatures, temp_free_energies)
    melt_T = interp_temps[np.argmin(np.abs(dg_interps))]

    fig = go.Figure()
    fig.add_scatter(x=temperatures, y=dF_bulk, mode='lines', name='Bulk free energy difference')
    fig.add_scatter(x=temperatures, y=dG_T, mode='lines', name='Loop Free Energy Difference')
    fig.show(renderer='browser')

    fig = go.Figure(go.Scatter(x=temperatures, y=temp_free_energies,
                               mode='lines+markers', showlegend=False, marker_color='blue'))
    fig.add_scatter(x=interp_temps, y=dg_interps,
                    mode='lines', showlegend=False, marker_color='green')
    fig.add_scatter(x=[melt_T], y=[0], text=[f"{melt_T:.0f} K"],
                    marker_color='red', marker_size=20, showlegend=False, mode='markers+text',
                    textposition='top right')
    fig.add_scatter(x=temperatures, y=np.zeros_like(temperatures), mode='lines', showlegend=False, marker_color='black')
    fig.update_layout(xaxis_title='Temperature (K)', yaxis_title="L/S Free Energy Difference")
    fig.show(renderer='browser')

    return None
