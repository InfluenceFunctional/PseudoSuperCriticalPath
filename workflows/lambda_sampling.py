from pathlib import Path
import os
from shutil import copy
from typing import Optional

import numpy as np


def lambda_runs(run_type: str,
                structure_name: str,
                reference_temperature: int,
                runs_directory: Path,
                num_restart_runs: int,
                sampling_time: int,  # in fs
                restart_sampling_time: int,  # in fs
                num_stage_two_restarts: Optional[int] = None,
                stage_one_sampling_dir: Optional[str] = None,
                ):
    pscp_dir = Path(__file__).parent.parent.resolve()
    slurm_file = pscp_dir.joinpath(Path('common').joinpath(Path("sub_job.slurm")))

    structure_directory = Path(runs_directory).joinpath(structure_name)

    if not os.path.exists(structure_directory):
        assert False, "Structure directory does not exist - missing bulk runs"

    stage_directory = structure_directory.joinpath(run_type)
    if not os.path.exists(stage_directory):
        assert False, "Missing lambda generation trajectory directory!"

    sampling_dir = Path('gen_run')
    gen_directory = stage_directory.joinpath(sampling_dir)
    if run_type == 'stage_one':
        gen_directory = gen_directory.joinpath(stage_one_sampling_dir)

    run_md_name = run_type + '_restart_' + 'run_MD.lmp'
    run_md_path = Path(__file__).parent.resolve().joinpath(run_md_name)

    restart_files = os.listdir(gen_directory)
    restart_files = [file for file in restart_files if '.restart.' in file]
    num_lambdas = len(restart_files)

    lambdas_to_restart = np.linspace(1/num_lambdas, 1, num_restart_runs)
    all_restart_steps = np.sort(np.asarray([int(file.split('.')[-1]) for file in restart_files]))
    #all_restart_steps = np.linspace(sampling_time // num_lambdas, sampling_time, num_lambdas).astype(int)
    steps_to_restart = all_restart_steps[np.linspace(0, num_lambdas - 1, num_restart_runs).astype(int)]

    num_steps = int(sampling_time)
    if num_stage_two_restarts is not None:
        restart_time = num_steps // num_stage_two_restarts

    restarts_dir = stage_directory.joinpath('restart_runs')
    if not os.path.exists(restarts_dir):
        os.mkdir(restarts_dir)

    for lambda_ind in range(num_restart_runs):
        # index of the step to restart
        restart_step = steps_to_restart[lambda_ind]
        current_lambda = lambdas_to_restart[lambda_ind]

        # lambda value of the step to restart
        restart_step_string = str(restart_step)
        lambda_restart_dir = restarts_dir.joinpath('lambda_' + f"{current_lambda:.3g}".replace('.','-'))
        gen_file_path = gen_directory.joinpath(f'{run_type}.restart.{restart_step_string}')

        if not os.path.exists(lambda_restart_dir):
            os.mkdir(lambda_restart_dir)

        # copy in the relevant files
        copy(gen_file_path, lambda_restart_dir)

        copy(gen_directory.joinpath(Path('system.in.settings.hybrid_overlay')),
             lambda_restart_dir)

        with (open(gen_directory.joinpath(Path('create_atoms.txt')), "r") as read,
              open(lambda_restart_dir.joinpath(Path('coeffs.txt')), "w") as write):
            for line in read:
                if line.startswith("mass") or line.startswith("create_atoms"):
                    continue
                write.write(line)

        with (open(run_md_path, "r") as read,
              open(lambda_restart_dir.joinpath(Path('run_MD.lmp')), "w") as write):
            text = read.read()
            text = text.replace("_T_SAMPLE", str(reference_temperature))
            text = text.replace("_N_STEPS", str(restart_sampling_time))
            text = text.replace("_LAMBDA", str(current_lambda))
            text = text.replace("_RESTART_FILE", f"{run_type}.restart.{restart_step_string}")
            if run_type == 'stage_two':
                text = text.replace("_STEPS_RESTART", str(restart_time))

            write.write(text)

        copy(slurm_file, lambda_restart_dir)

        d = os.getcwd()
        os.chdir(lambda_restart_dir)
        os.system("sbatch sub_job.slurm")
        os.chdir(d)
