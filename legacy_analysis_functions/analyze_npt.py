import matplotlib.pyplot as plt
import numpy as np
import subprocess


def main():
    temperature_step = 10
    minimum_temperature = 300
    maximum_temperature = 500
    temperatures = np.arange(minimum_temperature, maximum_temperature + temperature_step, temperature_step)

    for directory in ["npt_simulations/fluid", "npt_simulations/solid"]:
        kappa_values = {"1": [], "2": []}
        kappa_errors = {"1": [], "2": []}
        masses = {}
        first = True
        for temperature in temperatures:
            print(f"Analyzing directory {directory}/T_{temperature}")
            subprocess.run(f"python analyze.py {directory}/T_{temperature}/screen.log "
                           f"{directory}/T_{temperature}/tmp.out {directory}/T_{temperature}/analysis "
                           "--ramp_time=2000000", check=True, shell=True)
            if directory == "npt_simulations/fluid":
                topology_file = "bulk_fluid/0/new_system.data"
            else:
                topology_file = "bulk_solid/0/new_system.data"
            subprocess.run(
                f"python analyze_rdf.py {directory}/T_{temperature}/traj.dump "
                f"{topology_file} {directory}/T_{temperature}/analysis "
                "--ramp_steps=2000000 --dump_interval=10000", check=True, shell=True)

            if directory == "npt_simulations/solid":
                subprocess.run(
                    f"python analyze_fluctuations.py {directory}/T_{temperature}/traj.dump "
                    f"{topology_file} {directory}/T_{temperature}/analysis {temperature} "
                    "--ramp_steps=2000000 --dump_interval=10000 --total_steps=4000000", check=True, shell=True)
                with open(f"{directory}/T_{temperature}/analysis/kappa.txt", "r") as file:
                    for line in file:
                        if line.startswith("#"):
                            continue
                        split_line = line.split()
                        assert split_line[0] in kappa_values
                        if first:
                            assert split_line[0] not in masses
                            masses[split_line[0]] = float(split_line[1])
                        else:
                            assert float(split_line[1]) == masses[split_line[0]]
                        kappa_values[split_line[0]].append(float(split_line[2]))
                        kappa_errors[split_line[0]].append(float(split_line[3]))
                first = False
        if directory == "npt_simulations/solid":
            plt.figure()
            for t in kappa_values:
                plt.errorbar(temperatures, kappa_values[t], yerr=kappa_errors[t],
                             label=f"type {t} with mass {masses[t]} (mean={np.mean(kappa_values[t])})")
            plt.legend()
            plt.xlabel("temperature $T (K)$")
            plt.ylabel("fitted kappa (kcal/mol)")
            plt.savefig(f"{directory}/kappa.pdf", bbox_inches="tight")
            plt.close()
        else:
            assert kappa_values == {"1": [], "2": []}
            assert kappa_errors == {"1": [], "2": []}
            assert masses == {}


if __name__ == '__main__':
    main()
