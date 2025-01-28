import os
from pathlib import Path
from shutil import copy
from typing import Optional


def md_sampling(run_type: str,
                structure_name: str,
                temperatures_list: list,
                runs_directory: Path,
                sampling_time: int,  # in fs
                equilibration_time: Optional[int] = None,  # in fs
                ):
    structure_directory = Path(runs_directory).joinpath(structure_name)

    pscp_dir = Path(__file__).parent.parent.resolve()
    slurm_file = pscp_dir.joinpath(Path('common').joinpath("sub_job.slurm"))

    run_md_name = run_type.lower() + '_run_MD.lmp'
    run_md_path = Path(__file__).parent.resolve().joinpath(run_md_name)

    stage_directory = structure_directory.joinpath(f'{run_type.lower}_simulations')
    if not os.path.exists(stage_directory):
        os.mkdir(stage_directory)

    for phase in ['solid', 'fluid']:
        phase_directory = stage_directory.joinpath(Path(phase))
        if not os.path.exists(phase_directory):
            os.mkdir(phase_directory)

        for temp in temperatures_list:
            run_dir = phase_directory.joinpath(str(int(temp)))
            if not os.path.exists(run_dir):
                os.mkdir(run_dir)

            '''copy relevant files from source dir'''
            if 'npt' in run_type.lower():  # NPT runs restart from initial sampling
                old_phase_directory = structure_directory.joinpath(Path(f'bulk_{phase}/0'))
                copy(old_phase_directory.joinpath('cluster_1x1x1_equi.restart'),
                     run_dir.joinpath('cluster_init.restart'))
            elif 'nvt' in run_type.lower():  # NVT runs restart from NPT runs
                old_phase_directory = structure_directory.joinpath(Path(f'npt_simulations/{phase}/T_{int(str(temp))}'))
                copy(old_phase_directory.joinpath('cluster_finished.restart'),
                     run_dir.joinpath('cluster_init.restart'))
            else:
                assert False, f'{run_type} is not a valid run type'

            copy(old_phase_directory.joinpath('system.in.settings'),
                 run_dir.joinpath('system.in.settings'))
            copy(slurm_file,
                 run_dir.joinpath("sub_job.slurm"))

            with open(run_md_path, "r") as read, open(run_dir.joinpath('run_MD.lmp'), "w") as write:
                text = read.read()
                text = text.replace("_T_SAMPLE", str(temp))
                text = text.replace("_N_STEPS", str(sampling_time))

                if equilibration_time is not None:
                    text = text.replace("_N_EQUIL_STEPS", str(equilibration_time))
                    text = text.replace("#_EQUILIBRATE", '')

                if 'nvt' in run_type.lower():
                    text = text.replace("#_NVT", '')

                if 'npt' in run_type.lower():
                    text = text.replace("#_NPT", '')

                write.write(text)

            copy(slurm_file, run_dir.joinpath(Path("sub_job.slurm")))

            d = os.getcwd()
            os.chdir(run_dir)
            os.system("sbatch sub_job.slurm")
            os.chdir(d)


"""
simulation types

-: 'bulk_fluid'
    - single run
    - initialized from e2emolmats
    - NPT, 700K, 2ns, 2ns equil, 1atm iso
    - 200 prints
-: 'bulk_solid'
    - single run
    - initialized from e2emolmats
    - NPT, 100K, 2ns sample, 2ns equil, 1atm iso
    - 200 prints
-: 'nvt'
    - single run for 'fluid' and 'solid'
    - NVT, 400K, 2ns
-: 'npt'
    - apparently generated from bulk runs
    - set for 'fluid' and set for 'solid'
    - 300-500K
    - generated from setup file
    - 2ns equil, 2ns sample, npt
    
"""
