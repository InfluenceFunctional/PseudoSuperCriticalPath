import os
from shutil import copy


def main():
    reference_temperature = 400

    fluid_directory = f"../npt_simulations/fluid/T_{reference_temperature}/"
    solid_directory = f"../npt_simulations/solid/T_{reference_temperature}/"
    copy_filenames = ["system.in.settings"]
    number_steps = 2000000
    restart_filename = "cluster_equi.restart"
    slurm_filename = "sub_job.slurm"

    for t, ref_directory in zip(("fluid", "solid"), (fluid_directory, solid_directory)):
        try:
            os.mkdir(t)
        except FileExistsError:
            pass

        try:
            os.mkdir(f"{t}/T_{reference_temperature}")
        except FileExistsError:
            pass

        for copy_filename in copy_filenames:
            copy(f"{ref_directory}{copy_filename}", f"{t}/T_{reference_temperature}/{copy_filename}")
        copy(f"{ref_directory}{restart_filename}", f"{t}/T_{reference_temperature}/{restart_filename}")

        with open("run_MD.lmp", "r") as read, open(f"{t}/T_{reference_temperature}/run_MD.lmp", "w") as write:
            text = read.read()
            text = text.replace("_T_SAMPLE", str(reference_temperature))
            text = text.replace("_N_STEPS", str(number_steps))
            text = text.replace("_RESTART_FILE", restart_filename)
            if t == "solid":
                text = text.replace("#_NPT_TRI", "")
            write.write(text)

        copy(f"{ref_directory}{slurm_filename}", f"{t}/T_{reference_temperature}/{slurm_filename}")

        d = os.getcwd()
        os.chdir(f"{t}/T_{reference_temperature}")
        os.system("sbatch sub_job.slurm")
        os.chdir(d)


if __name__ == '__main__':
    main()
