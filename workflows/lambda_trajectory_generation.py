from pathlib import Path
import os
from shutil import copy

from workflows.utils import generate_system_settings_file, generate_create_atoms, find_hydrogen_type


def gen_run(run_type: str,
            structure_name: str,
            reference_temperature: int,
            runs_directory: Path,
            num_restarts: int,
            sampling_time: int,  # in fs
            ):

    temperature = float(reference_temperature)

    structure_directory = Path(runs_directory).joinpath(structure_name)

    if not os.path.exists(structure_directory):
        assert False, "Structure directory does not exist - missing bulk runs"

    stage_directory = structure_directory.joinpath(Path(run_type))
    if not os.path.exists(stage_directory):
        os.mkdir(stage_directory)

    (init_settings_path,
     new_settings_path, mean_positions_path,
     old_data_file_path, old_settings_path,
     slurm_file, solid_directory, topology_file_path) = process_paths(
        reference_temperature, structure_directory)
    run_md_name = run_type + '_gen_' + 'run_MD.lmp'
    run_md_path = Path(__file__).parent.resolve().joinpath(run_md_name)

    # if run_type == 'stage_one':
    #     well_widths_angstrom_squared = [0.01, 0.05, 0.1]
    #     well_depths = [0.1, 0.5, 1.0]
    # else:
    well_widths_angstrom_squared = [1]
    well_depths = [1]

    num_steps = int(sampling_time)
    restart_time = num_steps // num_restarts

    for well_width in well_widths_angstrom_squared:
        for well_depth in well_depths:
            # if run_type == 'stage_one':
            #     run_name = Path(f"well_width_{well_width}_well_depth_{well_depth}")
            # else:
            run_dir = stage_directory.joinpath(Path('gen_sampling'))

            if not os.path.exists(run_dir):
                print(f'making run {run_dir}')
                os.mkdir(run_dir)
                # print(f'{run_dir} already exists! Skipping. Delete or move existing run to retry.')
                # continue

            copy(old_data_file_path, run_dir.joinpath(Path("cluster_finished.data")))
            copy(init_settings_path, run_dir.joinpath(Path("new_system.in.init")))

            generate_system_settings_file(run_dir.joinpath(Path(new_settings_path)), old_settings_path, init_settings_path)

            number_new_types = generate_create_atoms(
                run_dir.joinpath("create_atoms.txt"),
                mean_positions_path,
                find_hydrogen_type(topology_file_path),
                well_width,
                well_depth
            )

            with open(run_md_path, "r") as read, open(run_dir.joinpath('run_MD.lmp'), "w") as write:
                text = read.read()
                text = text.replace("_T_SAMPLE", str(temperature))
                text = text.replace("_N_STEPS", str(num_steps))
                text = text.replace("_N_NEW_TYPES", str(number_new_types))
                text = text.replace("_STEPS_RESTART", str(restart_time))
                write.write(text)

            copy(slurm_file, run_dir.joinpath(Path("sub_job.slurm")))

            d = os.getcwd()
            os.chdir(run_dir)
            os.system("sbatch sub_job.slurm")
            os.chdir(d)


def process_paths(reference_temperature, head_dir):
    solid_directory = head_dir.joinpath(
        Path(f'nvt_simulations/solid/{int(reference_temperature)}')
    )
    init_settings = head_dir.joinpath(
        Path('bulk_solid/0/new_system.in.init')
    )
    topology_file = head_dir.joinpath(
        Path('bulk_solid/0/new_system.data')
    )
    old_settings = solid_directory.joinpath(
        Path('system.in.settings')
    )
    nvt_mean_positions = solid_directory.joinpath(
        Path('analysis/mean_positions.xyz')
    )
    old_data_file = solid_directory.joinpath(
        Path('cluster_finished.data')
    )
    new_settings = solid_directory.joinpath(
        Path("system.in.settings.hybrid_overlay")
    )

    pscp_dir = Path(__file__).parent.parent.resolve()
    slurm_file = pscp_dir.joinpath(Path('common').joinpath(Path("sub_job.slurm")))

    return init_settings, new_settings, nvt_mean_positions, old_data_file, old_settings, slurm_file, solid_directory, topology_file
