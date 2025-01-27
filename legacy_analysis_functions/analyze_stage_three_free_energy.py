import os
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymbar
from openmm import unit
from pymbar import timeseries
from analyze import compose_row_function
from pathlib import Path
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx} \DeclareSIUnit{\atm}{atm} \DeclareSIUnit{\cal}{cal}"
})
plt.rcParams.update(plt.rcParamsDefault)


def main():
    # lambda_step = 0.01
    # lambdas = np.linspace(lambda_step, 1, int(1/lambda_step))
    lmax = 1
    mbar_init = "BAR"  # "BAR" "zeros"
    num_samples_list = [1000]
    directory = Path("stage_three/runs_at_each_lambda")

    lambda_dirs = os.listdir(directory)
    lambda_dirs = [dir for dir in lambda_dirs if "lambda_" in dir]
    lambda_steps = np.sort([float(dir.split('_')[-1]) for dir in lambda_dirs]).astype(int)

    temperature = 400  # kelvin
    pressure = 1.0  # atm
    number_molecules = np.loadtxt("npt_simulations/fluid/T_300/tmp.out", skiprows=3, max_rows=1, dtype=int)[1]

    if os.path.isfile(f"{directory}/flat_coul_series.txt") and os.path.isfile(f"{directory}/n_k.txt"):
        coul_series = np.loadtxt(f"{directory}/flat_coul_series.txt")
        lj_series = np.loadtxt(f"{directory}/flat_lj_series.txt")
        gauss_series = np.loadtxt(f"{directory}/flat_gauss_series.txt")
        scale_series = np.loadtxt(f"{directory}/scale_series.txt")
        n_k = np.loadtxt(f"{directory}/n_k.txt", dtype=int)
        assert len(coul_series) == np.sum(n_k)
    else:
        print("Reading timeseries.")
        coul_series = np.empty(0)
        lj_series = np.empty(0)
        gauss_series = np.empty(0)
        n_k = np.empty(0, dtype=int)
        scale_series = np.zeros((len(lambda_steps), 4))
        for ind, lmbd in enumerate(tqdm(lambda_steps)):
            run_directory = Path(f"{directory}/lambda_{lmbd}")
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

            intra_energy = data['E_mol'].to_numpy() * unit.kilocalorie_per_mole
            coul_energy = (data['c_coul'] + data['E_long']/data['v_scale_coulomb']).to_numpy() * unit.kilocalorie_per_mole
            gauss_energy = data['c_gauss'].to_numpy() * unit.kilocalorie_per_mole
            lj_energy = data['c_lj'].to_numpy() * unit.kilocalorie_per_mole
            scale_series[ind, :] = [data['v_scale_coulomb'][0], data['v_scale_lj'][0], data['v_scale_gauss'][0], lmbd]
            data_full = (
                    coul_energy + gauss_energy + lj_energy
            )
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
        np.savetxt(f"{directory}/flat_coul_series.txt", coul_series)
        np.savetxt(f"{directory}/flat_lj_series.txt", lj_series)
        np.savetxt(f"{directory}/flat_gauss_series.txt", gauss_series)
        np.savetxt(f"{directory}/n_k.txt", n_k, fmt="%i")
        np.savetxt(f"{directory}/scale_series.txt", scale_series)
        assert len(coul_series) == np.sum(n_k)
        print("Finished reading timeseries.")

    lambda_steps = scale_series[:, 3]
    # convert from lambda steps to lambdas
    lambdas = lambda_steps / lambda_steps[-1]
    lambdas = lambdas[lambdas <= lmax]

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

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=lambdas, y=c_ens / number_molecules, name='Coul Energy')
    fig.add_scatter(x=lambdas, y=lj_ens / number_molecules, name='LJ Energy')
    fig.add_scatter(x=lambdas, y=g_ens / number_molecules, name='Gauss Energy')
    fig.show(renderer='browser')

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

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    k_vals = np.linspace(0, len(lambdas) - 2, 25).astype(int)
    batch = np.repeat(np.arange(len(lambdas)), n_k, axis=0)

    fig = make_subplots(rows=5, cols=5, subplot_titles=[f"Lambda = {k / 100:.2g}" for k in k_vals])
    for ind in range(25):
        row = ind // 5 + 1
        col = ind % 5 + 1
        k_ind = k_vals[ind]

        fig.add_histogram(x=ukn[k_ind, batch == k_ind], nbinsx=50, row=row, col=col, name="Current",
                          legendgroup="Current", marker_color='red', showlegend=ind == 0)
        fig.add_histogram(x=ukn[k_ind + 1, batch == k_ind + 1], nbinsx=50, row=row, col=col, name="Next",
                          legendgroup="Next", marker_color='blue', showlegend=ind == 0)
    fig.show()

    means = np.zeros(len(lambdas))
    stds = np.zeros(len(lambdas))
    for ind in range(len(lambdas)):
        means[ind] = ukn[ind, batch == ind].mean()
        stds[ind] = ukn[ind, batch == ind].std()
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=lambdas, y=means, mode='lines+markers', error_y=dict(type='data', array=stds**2, visible=True))).show(renderer='browser')

    import plotly.graph_objects as go
    #
    # fig = go.Figure()
    # batch = np.repeat(np.arange(len(lambdas)), n_k, axis=0)
    # overlap = np.zeros(len(lambdas) - 1)
    # for ind in range(len(lambdas) - 1):
    #     h1, b1 = np.histogram(ukn[ind, batch == ind], bins=50, density=True)
    #     h2, _ = np.histogram(ukn[ind + 1, batch == ind + 1], bins=50, density=True, range=[b1[0], b1[-1]])
    #
    #     h3, b1 = np.histogram(ukn[ind + 1, batch == ind + 1], bins=50, density=True)
    #     h4, _ = np.histogram(ukn[ind, batch == ind], bins=50, density=True, range=[b1[0], b1[-1]])
    #     overlap[ind] = (np.nan_to_num(np.sum(np.minimum(h1, h2))) + np.nan_to_num(np.sum(np.minimum(h3, h4)))) / 2
    #
    # fig.add_scatter(x=lambdas, y=overlap, name="Nearest-Neighbor Overlap")
    # fig.add_scatter(x=lambdas, y=np.diff(scale_series[:, 0]), name='Coul Slope')
    # fig.add_scatter(x=lambdas, y=np.diff(scale_series[:, 1]), name='LJ Slope')
    # fig.add_scatter(x=lambdas, y=np.diff(scale_series[:, 2]), name='Gauss Slope')
    # fig.show()

    # subsample ukns
    for max_samples in num_samples_list:
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

        mbar = pymbar.MBAR(new_ukn, new_nk, initialize=mbar_init)
        result = mbar.compute_free_energy_differences()

        # opt plot overlap matrix
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        overlap = mbar.compute_overlap()
        nn_overlap = np.array([overlap['matrix'][i, i+1] for i in range(len(overlap['matrix'])-1)])
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

        aa = 1

    def plot_run_thermo(lmbd):
        """
        Utility function to plot the thermo data for a particular run
        :param lmbd:
        :return:
        """
        directory = Path("stage_three/runs_at_each_lambda/")

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

        fig.show(renderer='browser')

        '''
        try to predict local overlaps
        '''

        num_lambdas = 100
        energy = data['PotEng']
        lambdas = np.linspace(0, 1, len(energy))

        temps = np.linspace(0.1, 1, 100)
        fit_mean = np.zeros(len(temps))
        fit_std = np.zeros(len(temps))
        fit_corr = np.zeros(len(temps))

        for t_ind, step_temp in enumerate(temps):
            lambda_steps = np.linspace(0, 1, num_lambdas)
            lambda_steps = ((lambda_steps ** step_temp) * (len(energy) - 1)).astype(int)  # rescale sampling window
            lambda_energies, lambda_widths, nn_overlaps, overlaps = compute_lambda_overlaps(energy, lambda_steps,
                                                                                         num_lambdas)
            fit_mean[t_ind] = nn_overlaps.mean()
            fit_std[t_ind] = nn_overlaps.std()
            fit_corr[t_ind] = np.corrcoef(lambda_steps[1:], nn_overlaps)[0,1]

        fig = go.Figure()
        fig.add_scatter(x=temps, y=fit_mean, name='mean overlaps')
        fig.add_scatter(x=temps, y=fit_std, name='overlap std')
        fig.add_scatter(x=temps, y=np.abs(fit_corr), name='correlation')
        fig.show(renderer='browser')

        '''best step is the one with minimal correlation'''
        step_temp = temps[np.argmin(np.abs(fit_corr))]
        lambda_steps = np.linspace(0, 1, num_lambdas)
        lambda_steps = ((lambda_steps ** step_temp) * (len(energy) - 1)).astype(int)  # rescale sampling window
        lambda_energies, lambda_widths, nn_overlaps, overlaps = compute_lambda_overlaps(energy, lambda_steps,
                                                                                        num_lambdas)

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
        fig.add_heatmap(z=overlaps,
                        row=1, col=2
                        )
        fig.add_scatter(x=lambdas[lambda_steps],
                        y=nn_overlaps,
                        showlegend=True,
                        name='Nearest-Neighbor Overlap',
                        row=1, col=3
                        )

        fig.show(renderer='browser')

    def compute_lambda_overlaps(energy, lambda_steps, num_lambdas):
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
                overlaps[ind1, ind2] = NormalDist(mu=e1, sigma=s1).overlap(NormalDist(mu=e2, sigma=s2))
        nn_overlaps = np.array(
            [overlaps[ind, ind - 1] for ind in range(1, num_lambdas)]
        )
        return lambda_energies, lambda_widths, nn_overlaps, overlaps


if __name__ == '__main__':
    main()
