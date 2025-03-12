from itertools import chain
from pathlib import Path
from typing import Callable, Iterable

import MDAnalysis


def generate_system_settings_file(new_system_settings: Path,
                                  old_system_settings: Path,
                                  init_settings: Path) -> None:
    pair_modify_mix = None

    # extract pair_modify mix mode
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
        # extract pair coefficients for lj/charmm/coul/long format
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

        # reconfigure pair coefficients into lj/charmm and coul/long/charm formats, with the appropriate pair mix
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


def generate_create_atoms(filename: Path,
                          mean_positions_file: Path,
                          hydrogen_type: int,
                          well_width_angstrom_squared: float,
                          well_depth: float) -> int:
    universe = MDAnalysis.Universe(mean_positions_file, in_memory=True, format="XYZ")
    old_types = set(int(atom.type) for atom in universe.atoms)
    max_type = max(old_types)
    heavy_atoms = universe.select_atoms(f"not type {hydrogen_type}")
    new_types = set(int(atom.type) + max_type for atom in heavy_atoms)

    # kappa_values = {}
    # with open(kappa_file, "r") as file:
    #     for line in file:
    #         if line.startswith("#"):
    #             continue
    #         split_line = line.split()
    #         assert len(split_line) == 4
    #         assert split_line[0] not in kappa_values
    #         _ = float(split_line[1])
    #         _ = float(split_line[3])
    #         kappa_values[int(split_line[0])] = float(split_line[2])
    # print(kappa_values)

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
                print(f"pair_coeff {old_type} {old_type + max_type} gauss "
                      f"{well_depth / well_width_angstrom_squared} {well_width_angstrom_squared}",
                      file=write_file)

        print("", file=write_file)

        print(f"group mobile type <> 1 {max_type}", file=write_file)
        print(f"group stat type <> {min(new_types)} {max(new_types)}", file=write_file)

    return len(new_types)


def yield_valid_rows(log_file: str) -> Iterable[tuple[int, int]]:
    start_index = None
    final_index = None
    with open(log_file, "r") as file:
        for line_index, line in enumerate(file):
            if line.strip().startswith("Step"):
                assert start_index is None
                start_index = line_index + 1
            if line.strip().startswith("Loop time"):
                assert start_index is not None
                assert final_index is None
                final_index = line_index - 1
                yield start_index, final_index
                start_index = None
                final_index = None


def compose_row_function(log_file: str, minimization: bool) -> Callable[[int], bool]:
    ranges = []
    for index, (start, end) in enumerate(yield_valid_rows(log_file)):
        if minimization and index == 0:
            continue
        ranges.append([start, end])
    # Include header.
    if ranges == []:
        raise ValueError("No valid rows found")
        print("Invalid screen.log file")
    else:
        ranges[0][0] -= 1

    # Remove overlaps.
    for i in range(1, len(ranges)):
        ranges[1][0] += 1

    def skip_row(row: int) -> bool:
        for s, e in ranges:
            if s <= row <= e:
                return False
        return True

    return skip_row
