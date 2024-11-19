import os
from shutil import copy


def main():
    reference_temperature = 400
    # 4.0 estimated from Mathematica script, 0.9 used by them, 0.09 reported in the paper.
    well_widths_angstrom_squared = [4.0, 0.9, 0.09]

    number_steps = 2000000
    number_restart_steps = 20000  # 100 restart files

    for well_width in well_widths_angstrom_squared:
        try:
            os.mkdir(f"well_width_{well_width}")
        except FileExistsError:
            pass

        for i in range(1, number_steps // number_restart_steps + 1):
            current_restart_step = i * number_restart_steps
            current_lambda = current_restart_step / number_steps

            try:
                os.mkdir(f"well_width_{well_width}/lambda_{current_lambda}")
            except FileExistsError:
                pass

            copy(f"../generate_restart_files_lambda/well_width_{well_width}/stage_one.restart.{current_restart_step}",
                 f"well_width_{well_width}/lambda_{current_lambda}/")

            copy(f"../generate_restart_files_lambda/well_width_{well_width}/system.in.settings.hybrid_overlay",
                 f"well_width_{well_width}/lambda_{current_lambda}/")

            with (open(f"../generate_restart_files_lambda/well_width_{well_width}/create_atoms.txt", "r") as read,
                  open(f"well_width_{well_width}/lambda_{current_lambda}/coeffs.txt", "w") as write):
                for line in read:
                    if line.startswith("mass") or line.startswith("create_atoms"):
                        continue
                    write.write(line)

            with (open("run_MD.lmp", "r") as read,
                  open(f"well_width_{well_width}/lambda_{current_lambda}/run_MD.lmp", "w") as write):
                text = read.read()
                text = text.replace("_T_SAMPLE", str(reference_temperature))
                text = text.replace("_N_STEPS", str(number_steps))
                text = text.replace("_LAMBDA", str(current_lambda))
                text = text.replace("_RESTART_FILE", f"stage_one.restart.{current_restart_step}")
                write.write(text)

            copy("sub_job.slurm", f"well_width_{well_width}/lambda_{current_lambda}/sub_job.slurm")

            d = os.getcwd()
            os.chdir(f"well_width_{well_width}/lambda_{current_lambda}")
            os.system("sbatch sub_job.slurm")
            os.chdir(d)


if __name__ == '__main__':
    main()
