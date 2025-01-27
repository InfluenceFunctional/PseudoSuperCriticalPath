import argparse
from functools import partial
import multiprocessing
import os
import pathlib
import shutil
import warnings
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis.rms as rms
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx} \DeclareSIUnit{\atm}{atm} \DeclareSIUnit{\cal}{cal}"
})


def probability_distribution(radii, kappa, beta):
    radii_squared = np.power(radii, 2)
    return np.power(beta * kappa / np.pi, 1.5) * 4.0 * np.pi * radii_squared * np.exp(-beta * kappa * radii_squared)


def transform_lammps_trajectory(trajectory_filename, number_of_total_frames, number_of_ignored_frames, ignore_types):
    # Fix problem of https://stackoverflow.com/questions/64261408
    from ovito import io
    from ovito.modifiers import DeleteSelectedModifier, SelectTypeModifier

    tmp_path = str(pathlib.Path(trajectory_filename).parent / "tmp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.mkdir(tmp_path)
    pipeline = io.import_file(trajectory_filename)
    assert pipeline.source.num_frames == number_of_total_frames
    if ignore_types is not None:
        pipeline.modifiers.append(SelectTypeModifier(types=set(ignore_types)))
        pipeline.modifiers.append(DeleteSelectedModifier())
    for i in range(number_of_ignored_frames, pipeline.source.num_frames):
        io.export_file(pipeline, f"{tmp_path}/tmp.{i}.data", "lammps/data", atom_style="full", frame=i)
    return pipeline.source.num_frames


def main():
    parser = argparse.ArgumentParser(description="Fit harmonic oscillators to the fluctuations of the heavy atoms.")
    parser.add_argument("trajectory", help="LAMMPS trajectory file", type=str)
    parser.add_argument("topology", help="LAMMPS topology file", type=str)
    parser.add_argument("output", help="output directory")
    parser.add_argument("temperature", help="temperature in Kelvin", type=float)
    parser.add_argument("--ramp_steps", help="number of time steps to ignore", type=int, default=2000000)
    parser.add_argument("--dump_interval", help="dump interval", type=int, default=10000)
    parser.add_argument("--total_steps", help="total number of time steps", type=int, default=4000000)
    parser.add_argument("--mean", help="store mean positions of heavy atoms", action="store_true")
    parser.add_argument("--ignore_types", help="ignore types in trajectory", type=int, nargs="+")
    args = parser.parse_args()

    tmp_path = str(pathlib.Path(args.trajectory).parent / "tmp")
    ramp_steps = args.ramp_steps
    dump_interval = args.dump_interval
    number_of_ignored_frames = ramp_steps // dump_interval
    total_frames = args.total_steps // dump_interval + 1
    print("NUMBER OF IGNORED FRAMES: ", number_of_ignored_frames)
    print("NUMBER OF TOTAL FRAMES: ", total_frames)

    process = multiprocessing.Process(target=transform_lammps_trajectory,
                                      args=(args.trajectory, total_frames, number_of_ignored_frames, args.ignore_types))
    process.start()
    process.join()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        universe = MDAnalysis.Universe(
            args.topology,
            [f"{tmp_path}/tmp.{i}.data" for i in range(number_of_ignored_frames, total_frames)],
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
    file = open(f"{args.output}/mean_positions.xyz", "w") if args.mean else None
    if file is not None:
        a, b, c, alpha, beta_ang, gamma = universe.dimensions
        lx = a
        xy = b * np.cos(np.deg2rad(gamma))
        xz = c * np.cos(np.deg2rad(beta_ang))
        ly = np.sqrt(b * b - xy * xy)
        yz = (b * c * np.cos(np.deg2rad(alpha)) - xy * xz) / ly
        lz = np.sqrt(c * c - xz * xz - yz * yz)
        print(len(universe.atoms), file=file)
        print(f'Lattice="{lx} 0.0 0.0 {xy} {ly} 0.0 {xz} {yz} {lz}" Properties="type:I:1:pos:R:3:id:I:1"', file=file)
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
            if file is not None:
                print(f"{atom.type} {mean_position[0]} {mean_position[1]} {mean_position[2]} {atom.id}", file=file)
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
    if file is not None:
        atoms = universe.select_atoms(f"type {hydrogen_type}")
        for atom in atoms:
            positions = np.array([atom.position for _ in universe.trajectory])
            mean_position = np.mean(positions, axis=0)
            print(f"{atom.type} {mean_position[0]} {mean_position[1]} {mean_position[2]} {atom.id}", file=file)
        file.close()

    with open(f"{args.output}/kappa.txt", "w") as file:
        print("# type\tmass\tkappa (kcal/mol)\tkappa_error (kcal/mol)", file=file)
        for type, mass, kappa, kappa_error in zip(types, masses, kappas, kappas_error):
            print(f"{type}\t{mass}\t{kappa}\t{kappa_error}", file=file)

    shutil.rmtree(tmp_path)


if __name__ == '__main__':
    main()
