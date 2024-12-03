import os
from shutil import copy


def main():
    reference_temperature = 400

    number_steps = 2000000
    number_restart_steps = 20000  # 100 restart files

    for i in range(1, number_steps // number_restart_steps + 1):
        current_restart_step = i * number_restart_steps
        current_lambda = current_restart_step / number_steps
        current_lambda_str = str(current_lambda).replace('.','-')

        with (open("run_MD_sample.lmp", "r") as read,
              open(f"lambda_{current_lambda_str}/run_MD.lmp", "w") as write):
            text = read.read()
            text = text.replace("_T_SAMPLE", str(reference_temperature))
            text = text.replace("_N_STEPS", str(number_steps))
            text = text.replace("_LAMBDA", str(current_lambda))
            text = text.replace("_STEPS_RESTART", str(number_restart_steps))
            text = text.replace("_RESTART_FILE", f" stage_two_lambda_equil.restart")
            write.write(text)

        copy("sub_job.slurm", f"lambda_{current_lambda_str}/sub_job.slurm")
        #
        # d = os.getcwd()
        # os.chdir(f"lambda_{current_lambda_str}")
        # os.system("sbatch sub_job.slurm")
        # os.chdir(d)


if __name__ == '__main__':
    main()
