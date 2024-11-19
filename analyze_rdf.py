import argparse
import os
os.environ['OVITO_GUI_MODE'] = '1'
import pathlib
import shutil
import warnings
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis.rdf as rdf
from ovito import io
from ovito.modifiers import DeleteSelectedModifier, SelectTypeModifier

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx} \DeclareSIUnit{\atm}{atm} \DeclareSIUnit{\cal}{cal}"
})


def main():
    parser = argparse.ArgumentParser(description="Calculate the radial distribution function of a LAMMPS trajectory.")
    parser.add_argument("trajectory", help="LAMMPS trajectory file", type=str)
    parser.add_argument("topology", help="LAMMPS topology file", type=str)
    parser.add_argument("output", help="output directory")
    parser.add_argument("--ramp_steps", help="number of time steps to ignore", type=int, default=2000000)
    parser.add_argument("--dump_interval", help="dump interval", type=int, default=10000)
    parser.add_argument("--ignore_types", help="ignore types in trajectory", type=int, nargs="+")
    args = parser.parse_args()

    ramp_steps = args.ramp_steps
    dump_interval = args.dump_interval
    number_of_ignored_frames = ramp_steps // dump_interval
    print("NUMBER OF IGNORED FRAMES: ", number_of_ignored_frames)

    tmp_path = str(pathlib.Path(args.trajectory).parent / "tmp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.mkdir(tmp_path)

    pipeline = io.import_file(args.trajectory)
    if args.ignore_types is not None:
        pipeline.modifiers.append(SelectTypeModifier(types=set(args.ignore_types)))
        pipeline.modifiers.append(DeleteSelectedModifier())
    for i in range(number_of_ignored_frames, pipeline.source.num_frames):
        io.export_file(pipeline, f"{tmp_path}/tmp.{i}.data", "lammps/data", atom_style="full", frame=i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        universe = MDAnalysis.Universe(
            args.topology,
            [f"{tmp_path}/tmp.{i}.data" for i in range(number_of_ignored_frames, pipeline.source.num_frames)],
            format="DATA", topology_format="DATA", in_memory=True)
        com_universe = MDAnalysis.Universe.empty(
            n_atoms=len(universe.residues), n_residues=len(universe.residues),
            atom_resindex=[i for i in range(len(universe.residues))],
            n_segments=1, trajectory=True, n_frames=len(universe.trajectory),
            velocities=False, forces=False)
        for _, __ in zip(universe.trajectory, com_universe.trajectory):
            com_universe.dimensions = universe.dimensions
            for residue_index, residue in enumerate(universe.residues):
                com_universe.atoms[residue_index].position = residue.atoms.center_of_mass()

    r1 = rdf.InterRDF(com_universe.atoms, com_universe.atoms, exclude_same="residue")
    r1.run()
    # r2 = rdf.InterRDF(universe.select_atoms("type 1"), universe.select_atoms("type 1"), exclude_same="residue")
    # r2.run()
    plt.figure()
    plt.plot(r1.results.bins, r1.results.rdf)
    # plt.plot(r2.results.bins, r2.results.rdf)
    plt.xlabel("distance $r / \\unit{\\angstrom}$")
    plt.ylabel("average radial distribution function $g(r)$")
    plt.savefig(f"{args.output}/rdf.pdf", bbox_inches="tight")
    plt.close()

    shutil.rmtree(tmp_path)


if __name__ == '__main__':
    main()
