import matplotlib.pyplot as plt
import numpy as np
import subprocess


def main():
    temperature = 400

    for directory in ["nvt_simulations/fluid", "nvt_simulations/solid"]:
        kappa_values = {"1": [], "2": []}
        kappa_errors = {"1": [], "2": []}
        masses = {}

        print(f"Analyzing directory {directory}/T_{temperature}")
        #subprocess.run(f"python analyze.py {directory}/T_{temperature}/screen.log "
        #               f"{directory}/T_{temperature}/tmp.out {directory}/T_{temperature}/analysis "
        #               "--ramp_time=0", check=True, shell=True)
        if directory == "nvt_simulations/fluid":
            topology_file = "bulk_fluid/0/new_system.data"
        else:
            topology_file = "bulk_solid/0/new_system.data"
        #subprocess.run(
        #    f"python analyze_rdf.py {directory}/T_{temperature}/traj.dump "
        #    f"{topology_file} {directory}/T_{temperature}/analysis "
        #    "--ramp_steps=0 --dump_interval=200", check=True, shell=True)

        if directory == "nvt_simulations/solid":
            subprocess.run(
                f"python analyze_fluctuations.py {directory}/T_{temperature}/traj.dump "
                f"{topology_file} {directory}/T_{temperature}/analysis {temperature} "
                "--ramp_steps=0 --dump_interval=200 --total_steps=2000000 --mean", check=True, shell=True)
            with open(f"{directory}/T_{temperature}/analysis/kappa.txt", "r") as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    split_line = line.split()
                    assert split_line[0] in kappa_values
                    assert split_line[0] not in masses
                    masses[split_line[0]] = float(split_line[1])
                    kappa_values[split_line[0]].append(float(split_line[2]))
                    kappa_errors[split_line[0]].append(float(split_line[3]))
            print(kappa_values)
            print(kappa_errors)


if __name__ == '__main__':
    main()
