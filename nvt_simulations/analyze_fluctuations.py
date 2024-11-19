import argparse
from functools import partial
import os
import shutil
import warnings
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis.rms as rms
import numpy as np
from ovito import io
from scipy.optimize import curve_fit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx} \DeclareSIUnit{\atm}{atm} \DeclareSIUnit{\cal}{cal}"
})


def probability_distribution(radii, kappa, beta):
    radii_squared = np.power(radii, 2)
    return np.power(beta * kappa / np.pi, 1.5) * 4.0 * np.pi * radii_squared * np.exp(-beta * kappa * radii_squared)


def main():
    parser = argparse.ArgumentParser(description="Fit harmonic oscillators to the fluctuations of the heavy atoms.")
    parser.add_argument("trajectory", help="LAMMPS trajectory file", type=str)
    parser.add_argument("topology", help="LAMMPS topology file", type=str)
    parser.add_argument("output", help="output directory")
    parser.add_argument("temperature", help="temperature in Kelvin", type=float)
    parser.add_argument("--ramp_steps", help="number of time steps to ignore", type=int, default=2000000)
    parser.add_argument("--dump_interval", help="dump interval", type=int, default=10000)
    args = parser.parse_args()

    ramp_steps = args.ramp_steps
    dump_interval = args.dump_interval
    number_of_ignored_frames = ramp_steps // dump_interval

    os.mkdir("tmp")

    pipeline = io.import_file(args.trajectory)
    for i in range(number_of_ignored_frames, pipeline.source.num_frames):
        io.export_file(pipeline, f"tmp/tmp.{i}.data", "lammps/data", atom_style="full", frame=i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        universe = MDAnalysis.Universe(
            args.topology,
            [f"tmp/tmp.{i}.data" for i in range(number_of_ignored_frames, pipeline.source.num_frames)],
            format="DATA", topology_format="DATA", in_memory=True)

    assert all(len(residue.atoms) == len(universe.residues[0].atoms) for residue in universe.residues)
    assert all(list(atom.type for atom in residue.atoms) == list(atom.type for atom in universe.residues[0].atoms)
               for residue in universe.residues)
    assert all(list(atom.mass for atom in residue.atoms) == list(atom.mass for atom in universe.residues[0].atoms)
               for residue in universe.residues)
    type_mass_dict = {}
    for atom in universe.residues[0].atoms:
        if atom.type not in type_mass_dict:
            type_mass_dict[atom.type] = atom.mass
        else:
            assert type_mass_dict[atom.type] == atom.mass
    print(f"Detected atom types and masses: {type_mass_dict}")
    hydrogen_type = min(type_mass_dict, key=type_mass_dict.get)
    print(f"Detected hydrogen type: {hydrogen_type}")

    heavy_atoms = universe.select_atoms(f"not type {hydrogen_type}")
    rmsf = rms.RMSF(heavy_atoms, verbose=True).run()
    plt.figure()
    plt.scatter(heavy_atoms.types, rmsf.results.rmsf, s=2)
    plt.xlabel("atom type")
    plt.ylabel(r"root mean square fluctuation (\AA)")
    plt.savefig(f"{args.output}/rmsf.pdf", bbox_inches="tight")
    plt.close()

    # Inverse temperature in units of mol/kcal.
    beta = 1.0 / (1.987204259e-3 * args.temperature)
    print(f"Beta: {beta}")
    kappas = []
    kappas_error = []
    types = []
    masses = []
    for type in type_mass_dict:
        if type == hydrogen_type:
            continue
        print(f"Considering type {type}.")
        plt.figure()
        all_radii = np.empty(0)
        rmsf_mean = np.mean(rmsf.results.rmsf[heavy_atoms.types == type])
        atoms = universe.select_atoms(f"type {type}")
        for atom in atoms:
            positions = np.array([atom.position for _ in universe.trajectory])
            mean_position = np.mean(positions, axis=0)
            positions -= mean_position
            radii = np.linalg.norm(positions, axis=1)
            all_radii = np.append(all_radii, radii)
            hist, bin_edges = np.histogram(radii, bins=50, range=(0, 5 * rmsf_mean), density=True)
            plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist, alpha=0.5, color="C0")
        hist, bin_edges = np.histogram(all_radii, bins=50, range=(0, 5 * rmsf_mean), density=True)
        plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist, color="k")

        # noinspection PyTupleAssignmentBalance
        popt, pcov = curve_fit(partial(probability_distribution, beta=beta),
                               (bin_edges[:-1] + bin_edges[1:]) / 2, hist,
                               p0=[4.0 / (np.pi * beta * rmsf_mean * rmsf_mean)])
        perr = np.sqrt(np.diag(pcov))
        assert len(popt) == 1
        print(f"Result from mean: {4.0 / (np.pi * beta * rmsf_mean * rmsf_mean)}")
        print(f"Result from fit: {popt[0]} +- {perr[0]}")
        plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2.0,
                 probability_distribution((bin_edges[:-1] + bin_edges[1:]) / 2.0, popt[0], beta), 'r-')
        plt.xlabel("radius (\AA)")
        plt.ylabel("probability density")
        plt.savefig(f"{args.output}/radius_distribution_type_{type}.pdf", bbox_inches="tight")
        plt.close()
        kappas.append(popt[0])
        kappas_error.append(perr[0])
        types.append(type)
        masses.append(type_mass_dict[type])

    with open(f"{args.output}/kappa.txt", "w") as file:
        print("# type\tmass\tkappa (kcal/mol)\tkappa_error (kcal/mol)", file=file)
        for type, mass, kappa, kappa_error in zip(types, masses, kappas, kappas_error):
            print(f"{type}\t{mass}\t{kappa}\t{kappa_error}", file=file)

    shutil.rmtree("tmp")


if __name__ == '__main__':
    main()
