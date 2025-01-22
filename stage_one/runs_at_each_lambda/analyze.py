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
