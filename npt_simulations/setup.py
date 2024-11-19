import os
from shutil import copy
import numpy as np


def main():
    temperature_step = 10
    minimum_temperature = 300
    maximum_temperature = 500

    fluid_directory = "../bulk_fluid/0/"
    solid_directory = "../bulk_solid/0/"
    copy_filenames = ["system.in.settings"]
    number_steps = 2000000
    number_equilibration_steps = 2000000
    restart_filename = "cluster_1x1x1_equi.restart"
    slurm_filename = "sub_job.slurm"

    starting_temperature_fluid = 700
    starting_temperature_solid = 100

    temperatures = np.arange(minimum_temperature, maximum_temperature + temperature_step, temperature_step)

    for t, ref_directory, t_init in zip(("fluid", "solid"),
                                        (fluid_directory, solid_directory),
                                        (starting_temperature_fluid, starting_temperature_solid)):
        try:
            os.mkdir(t)
        except FileExistsError:
            pass

        for index, temperature in enumerate(temperatures):
            try:
                os.mkdir(f"{t}/T_{temperature}")
            except FileExistsError:
                pass

            for copy_filename in copy_filenames:
                copy(f"{ref_directory}{copy_filename}", f"{t}/T_{temperature}/{copy_filename}")
            copy(f"{ref_directory}{restart_filename}", f"{t}/T_{temperature}/{restart_filename}")

            with open("run_MD.lmp", "r") as read, open(f"{t}/T_{temperature}/run_MD.lmp", "w") as write:
                text = read.read()
                text = text.replace("_T_SAMPLE", str(temperature))
                text = text.replace("_T_INIT", str(t_init))
                text = text.replace("_RND", str(index + 1))
                text = text.replace("_N_STEPS", str(number_steps))
                text = text.replace("_N_EQUIL", str(number_equilibration_steps))
                text = text.replace("_RESTART_FILE", restart_filename)
                if t == "fluid":
                    text = text.replace("#_NPT_ISO", "")
                else:
                    assert t == "solid"
                    text = text.replace("#_NPT_TRI", "")
                write.write(text)

            copy(f"{ref_directory}{slurm_filename}", f"{t}/T_{temperature}/{slurm_filename}")

            d = os.getcwd()
            os.chdir(f"{t}/T_{temperature}")
            os.system("sbatch sub_job.slurm")
            os.chdir(d)


if __name__ == '__main__':
    main()
