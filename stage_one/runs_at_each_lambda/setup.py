import os
from shutil import copy

import numpy as np
from pathlib import Path
from analyze_stage_one_generation_run import extract_lambda_spacing


def main():
    reference_temperature = 400

    number_steps = 2000000
    number_restart_steps = 2000  # 100 restart files
    dir_name = 'well_width_0.05_well_depth_0.1'
    gen_path = Path('../generate_restart_files_lambda').joinpath(Path(dir_name))
    restart_paths = os.listdir(gen_path)
    restart_paths = [p for p in restart_paths if 'stage_one.restart' in p]

    steps_to_restart = extract_lambda_spacing(100,
                                              len(restart_paths),
                                              gen_path,
                                              buffer=10,
                                              )

    steps_to_restart[0] = 1
    steps_to_restart[-1] = 999
    print(f"running at lambda points {steps_to_restart}")
    restart_steps = np.linspace(number_steps // 1000, number_steps, 1000).astype(int)

    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    for i in range(len(steps_to_restart)):
        current_restart_step = steps_to_restart[i]
        current_lambda = float(current_restart_step / 1000)
        restart_step = restart_steps[current_restart_step]
        restart_step_string = str(restart_step)

        try:
            os.mkdir(f"{dir_name}/lambda_{restart_step_string}")
        except FileExistsError:
            pass

        copy(f"../generate_restart_files_lambda/{dir_name}/stage_one.restart.{restart_step_string}",
             f"{dir_name}/lambda_{restart_step_string}/")

        copy(f"../generate_restart_files_lambda/{dir_name}/system.in.settings.hybrid_overlay",
             f"{dir_name}/lambda_{restart_step_string}/")

        with (open(f"../generate_restart_files_lambda/{dir_name}/create_atoms.txt", "r") as read,
              open(f"{dir_name}/lambda_{restart_step_string}/coeffs.txt", "w") as write):
            for line in read:
                if line.startswith("mass") or line.startswith("create_atoms"):
                    continue
                write.write(line)

        with (open("run_MD.lmp", "r") as read,
              open(f"{dir_name}/lambda_{restart_step_string}/run_MD.lmp", "w") as write):
            text = read.read()
            text = text.replace("_T_SAMPLE", str(reference_temperature))
            text = text.replace("_N_STEPS", str(number_steps))
            text = text.replace("_LAMBDA", str(current_lambda))
            text = text.replace("_RESTART_FILE", f"stage_one.restart.{restart_step_string}")
            write.write(text)

        copy("sub_job.slurm", f"{dir_name}/lambda_{restart_step_string}/sub_job.slurm")

        d = os.getcwd()
        os.chdir(f"{dir_name}/lambda_{restart_step_string}")
        os.system("sbatch sub_job.slurm")
        os.chdir(d)


if __name__ == '__main__':
    main()
