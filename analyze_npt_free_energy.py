import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymbar
from openmm import unit
from pymbar import timeseries
from analyze import compose_row_function


plt.rcParams.update(plt.rcParamsDefault)

def main():
    temperature_step = 10  # Kelvin
    minimum_temperature = 300  # Kelvin
    maximum_temperature = 500  # Kelvin
    pressure = 1.0  # atm
    temperatures = np.arange(minimum_temperature, maximum_temperature + temperature_step, temperature_step)
    fluid_directory = "npt_simulations/fluid"
    solid_directory = "npt_simulations/solid"

    number_molecules = np.loadtxt("npt_simulations/fluid/T_300/tmp.out", skiprows=3, max_rows=1, dtype=int)[1]

    for directory_index, directory in enumerate((fluid_directory, solid_directory)):
        if os.path.isfile(f"{directory}/flat_data_series.txt") and os.path.isfile(f"{directory}/n_k.txt"):
            data_series = np.loadtxt(f"{directory}/flat_data_series.txt")
            n_k = np.loadtxt(f"{directory}/n_k.txt", dtype=int)
            assert len(data_series) == np.sum(n_k)
        else:
            print("Reading timeseries.")
            data_series = np.empty(0)
            n_k = np.empty(0, dtype=int)
            for temperature in temperatures:
                assert (np.loadtxt(f"{directory}/T_{temperature}/tmp.out", skiprows=3, max_rows=1, dtype=int)[1]
                        == number_molecules)
                log_file = f"{directory}/T_{temperature}/screen.log"
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
            np.savetxt(f"{directory}/flat_data_series.txt", data_series)
            np.savetxt(f"{directory}/n_k.txt", n_k, fmt="%i")
            assert len(data_series) == np.sum(n_k)
            print("Finished reading timeseries.")

        # eqn 2 in the paper
        ukn = np.zeros((len(temperatures), len(data_series)))
        for i, temperature in enumerate(temperatures):
            beta = (1.0 / (unit.BOLTZMANN_CONSTANT_kB
                           * (temperature * unit.kelvin)
                           * unit.AVOGADRO_CONSTANT_NA)).value_in_unit(unit.kilocalorie_per_mole ** (-1))
            ukn[i, :] = beta * data_series[:]

        mbar = pymbar.MBAR(ukn, n_k, initialize="BAR")
        result = mbar.compute_free_energy_differences()
        plt.errorbar(temperatures, result["Delta_f"][0, :] / number_molecules,
                     yerr=result["dDelta_f"][0, :] / number_molecules, label=directory,
                     color=f"C{directory_index}")
        entropy = -np.gradient(result["Delta_f"][0, :], temperature_step)
        heat_capacity = temperatures * np.gradient(entropy, temperature_step)
        plt.plot(temperatures, -np.gradient(result["Delta_f"][0, :], temperature_step),
                 color=f"C{directory_index}", linestyle="--")
        plt.plot(temperatures, heat_capacity, color=f"C{directory_index}", linestyle=":")

    plt.xlabel("Temperature (K)")
    plt.ylabel("Dimensionless free energy difference per molecule")
    plt.legend()
    plt.savefig("test.pdf", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    main()
