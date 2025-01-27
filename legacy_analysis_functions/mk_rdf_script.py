import os
from pathlib import Path
os.environ['OVITO_GUI_MODE'] = '1'
import shutil
import warnings
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis.rdf as rdf
from ovito import io
from ovito.modifiers import DeleteSelectedModifier, SelectTypeModifier

runs_dir = Path(r'D:\crystal_datasets\pscp\stage_one\generate_restart_files_lambda')
dirs = os.listdir(runs_dir)
runs_to_analyze = [run for run in dirs if 'well_width_' in run]
number_of_ignored_frames = 10

for run_dir in runs_to_analyze[:2]:
    trajectory = runs_dir.joinpath(run_dir + '/traj.dump')
    topology = "D:\crystal_datasets\pscp\stage_one_fixed\generate_restart_files_lambda\well_width_4.0\cluster_stage_one_generate.data"
    pipeline = io.import_file(trajectory)

    for i in range(number_of_ignored_frames, pipeline.source.num_frames):
        io.export_file(pipeline, f"tmp.{i}.data", "lammps/data", atom_style="full", frame=i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        universe = MDAnalysis.Universe(
        topology,
            [f"tmp.{i}.data" for i in range(number_of_ignored_frames, pipeline.source.num_frames)],
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
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=r1.results.bins, y=r1.results.rdf)
    fig.show(renderer='browser')

    aa = 1