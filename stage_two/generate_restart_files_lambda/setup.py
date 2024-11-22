from itertools import chain
import MDAnalysis
import os
from shutil import copy


def generate_system_settings_file(new_system_settings: str, old_system_settings: str, init_settings: str) -> None:
    pair_modify_mix = None
    with open(init_settings, "r") as file:
        pair_modify_found = False
        for line in file:
            if line.startswith("pair_modify"):
                assert not pair_modify_found
                pair_modify_found = True
                split_line = line.split()
                assert len(split_line) == 3
                assert split_line[0] == "pair_modify"
                assert split_line[1] == "mix"
                assert split_line[2] in ("geometric", "arithmetic", "sixthpower")
                pair_modify_mix = split_line[2]
    assert pair_modify_mix is not None

    with open(old_system_settings, "r") as file, open(new_system_settings, "w") as write_file:
        epsilons = {}
        sigmas = {}
        for line in file:
            if line.startswith("pair_coeff"):
                split_line = line.split()
                assert len(split_line) == 6
                assert split_line[0] == "pair_coeff"
                assert int(split_line[1]) > 0
                assert int(split_line[2]) > 0
                assert int(split_line[1]) == int(split_line[2])
                assert split_line[3] == "lj/charmm/coul/long"
                assert float(split_line[4]) > 0.0
                assert float(split_line[5]) > 0.0
                epsilons[int(split_line[1])] = float(split_line[4])
                sigmas[int(split_line[1])] = float(split_line[5])
            else:
                print(line, file=write_file, end="")

        assert len(epsilons) == len(sigmas)
        for t in ("lj/charmm", "coul/long/charmm"):
            for i in epsilons.keys():
                assert i in sigmas
                print(f"pair_coeff {i} {i} {t} {epsilons[i]} {sigmas[i]}", file=write_file)
                for j in epsilons.keys():
                    if not j > i:
                        continue
                    # See https://docs.lammps.org/pair_modify.html
                    if pair_modify_mix == "geometric":
                        mixed_epsilon = (epsilons[i] * epsilons[j]) ** 0.5
                        mixed_sigma = (sigmas[i] * sigmas[j]) ** 0.5
                    elif pair_modify_mix == "arithmetic":
                        mixed_epsilon = (epsilons[i] * epsilons[j]) ** 0.5
                        mixed_sigma = (sigmas[i] + sigmas[j]) / 2.0
                    else:
                        assert pair_modify_mix == "sixthpower"
                        mixed_epsilon = (2.0 * ((epsilons[i] * epsilons[j]) ** 0.5)
                                         * (sigmas[i] ** 3) * (sigmas[j] ** 3)) / ((sigmas[i] ** 6) + (sigmas[j] ** 6))
                        mixed_sigma = (((sigmas[i] ** 6) + (sigmas[j] ** 6)) / 2.0) ** (1.0 / 6.0)
                    print(f"pair_coeff {i} {j} {t} {mixed_epsilon} {mixed_sigma}", file=write_file)


def find_hydrogen_type(topology_file: str) -> int:
    mass_dictionary = {}
    with open(topology_file, "r") as file:
        in_mass_section = False
        for line in file:
            if line.strip() == "Masses":
                assert not in_mass_section
                in_mass_section = True
                continue

            if in_mass_section:
                if line.strip() == "":
                    continue
                if line.startswith("Atoms"):
                    assert in_mass_section
                    assert len(mass_dictionary) > 0
                    in_mass_section = False
                    continue
                uncommented_line = line.split("#")[0]
                split_line = uncommented_line.split()
                assert len(split_line) == 2
                assert int(split_line[0]) > 0
                assert float(split_line[1]) > 0.0
                mass_dictionary[int(split_line[0])] = float(split_line[1])
                continue
    hydrogen_type = min(mass_dictionary, key=mass_dictionary.get)
    print(f"Ignoring type {hydrogen_type} with mass {mass_dictionary[hydrogen_type]}.")
    return hydrogen_type


def generate_create_atoms(filename: str, mean_positions_file: str, hydrogen_type: int,
                          ) -> int:
    universe = MDAnalysis.Universe(mean_positions_file, in_memory=True, format="XYZ")
    old_types = set(int(atom.type) for atom in universe.atoms)
    max_type = max(old_types)
    heavy_atoms = universe.select_atoms(f"not type {hydrogen_type}")
    new_types = set(int(atom.type) + max_type for atom in heavy_atoms)

    with open(filename, "w") as write_file:
        for new_type in new_types:
            print(f"mass {new_type} 0.001", file=write_file)

        print("", file=write_file)

        for atom in heavy_atoms:
            print(f"create_atoms {int(atom.type) + max_type} single {atom.position[0]} {atom.position[1]} "
                  f"{atom.position[2]} remap yes units box", file=write_file)

        print("", file=write_file)

        for new_type in new_types:
            for t in chain(old_types, new_types):
                if t <= new_type:
                    print(f"pair_coeff {t} {new_type} none", file=write_file)

        print("", file=write_file)

        for old_type in old_types:
            if old_type != hydrogen_type:
                print(f"pair_coeff {old_type} {old_type + max_type} gauss ",
                      file=write_file)

        print("", file=write_file)

        print(f"group mobile type <> 1 {max_type}", file=write_file)
        print(f"group stat type <> {min(new_types)} {max(new_types)}", file=write_file)

    return len(new_types)


def extract_box_lengths(filename):
    with open(filename, "r") as file:
        for line in file:
            if 'xlo' in line:
                splitline = line.split()
                xlo = float(splitline[0])
                xhi = float(splitline[1])
            elif 'ylo' in line:
                splitline = line.split()
                ylo = float(splitline[0])
                yhi = float(splitline[1])

            elif 'zlo' in line:
                splitline = line.split()
                zlo = float(splitline[0])
                zhi = float(splitline[1])
                break

    return xhi-xlo, yhi-ylo, zhi-zlo


def main():
    reference_temperature = 400

    equil_fluid_data_file = "../../bulk_fluid/0/cluster_1x1x1_equi.data"
    dense_weak_fluid_directory = f"../../stage_one/generate_restart_files_lambda/well_width_4.0/"
    init_settings = "../../bulk_solid/0/new_system.in.init"
    init_ff = "../../stage_one/generate_restart_files_lambda/well_width_4.0/system.in.settings.hybrid_overlay"
    # topology_file = "../../bulk_solid/0/new_system.data"
    # old_settings = f"{fluid_directory}system.in.settings.hybrid_overlay"
    old_data_file = f"{dense_weak_fluid_directory}cluster_stage_one_generate.data"
    slurm_file = "sub_job.slurm"

    # new_settings = "system.in.settings.hybrid_overlay"
    # new_atoms = "create_atoms.txt"
    # 4.0 estimated from Mathematica script, 0.9 used by them, 0.09 reported in the paper.

    number_steps = 2000000
    number_restart_steps = 20000  # 100 restart files

    x_length, y_length, z_length = extract_box_lengths(equil_fluid_data_file)

    copy(old_data_file, f"cluster_stage_one_generate.data")
    copy(init_settings, f"new_system.in.init")
    copy(init_ff, f"system.in.settings.hybrid_overlay")

    # generate_system_settings_file(f"/{new_settings}", old_settings,
    #                               init_settings)
    # number_new_types = generate_create_atoms(f"/{new_atoms}", nvt_mean_positions,
    #                                          find_hydrogen_type(topology_file))

    with open("run_MD_template.lmp", "r") as read, open(f"run_MD.lmp", "w") as write:
        text = read.read()
        text = text.replace("_T_SAMPLE", str(reference_temperature))
        text = text.replace("_N_STEPS", str(number_steps))
        text = text.replace("_N_NEW_TYPES", str(2))
        text = text.replace("_STEPS_RESTART", str(number_restart_steps))
        text = text.replace("_X_LENGTH", str(x_length))
        text = text.replace("_Y_LENGTH", str(y_length))
        text = text.replace("_Z_LENGTH", str(z_length))

        write.write(text)

    copy(slurm_file, f"/sub_job.slurm")

    # d = os.getcwd()
    # os.chdir(f"")
    # os.system("sbatch sub_job.slurm")
    # os.chdir(d)


if __name__ == '__main__':
    main()
