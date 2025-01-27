import concurrent.futures
import subprocess
import matplotlib.pyplot as plt
import numpy as np


def analyze(directory, lambda_value):
    print(f"Analyzing directory {directory}/lambda_{lambda_value}")
    subprocess.run(f"python analyze.py {directory}/lambda_{lambda_value}/screen.log "
                   f"{directory}/lambda_{lambda_value}/tmp.out {directory}/lambda_{lambda_value}/analysis "
                   "--ramp_time=0", check=True, shell=True)
    topology_file = "bulk_solid/0/new_system.data"
    subprocess.run(
        f"python analyze_rdf.py {directory}/lambda_{lambda_value}/traj.dump "
        f"{topology_file} {directory}/lambda_{lambda_value}/analysis "
        "--ramp_steps=0 --dump_interval=200 --ignore_types 4 5", check=True, shell=True)
    temperature = 400
    subprocess.run(
        f"python analyze_fluctuations.py {directory}/lambda_{lambda_value}/traj.dump "
        f"{topology_file} {directory}/lambda_{lambda_value}/analysis {temperature} "
        "--ramp_steps=0 --dump_interval=10000 --total_steps=2000000 --ignore_types 4 5", check=True, shell=True)


def main():
    main_directory = "stage_one/runs_at_each_lambda"
    lambda_values = [i * 20000 / 2000000 for i in range(1, 2000000 // 20000 + 1)]

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        for directory in [f"{main_directory}/well_width_0.09", f"{main_directory}/well_width_0.9",
                          f"{main_directory}/well_width_4.0"]:
            for lambda_value in lambda_values:
                futures.append(executor.submit(analyze, directory, lambda_value))

    for future in concurrent.futures.as_completed(futures):
        assert future.done()
        exception = future.exception()
        if exception is not None:
            raise exception

    for directory in [f"{main_directory}/well_width_0.09", f"{main_directory}/well_width_0.9",
                      f"{main_directory}/well_width_4.0"]:
        with open(f"{directory}/kappa.txt", "w") as write:
            print("# lambda\ttype\tmass\tkappa (kcal/mol)\tkappa_error (kcal/mol)", file=write)
            for lambda_value in lambda_values:
                with open(f"{directory}/lambda_{lambda_value}/analysis/kappa.txt", "r") as file:
                    for line in file:
                        if line.startswith("#"):
                            continue
                        split_line = line.split()
                        assert len(split_line) == 4
                        print(f"{lambda_value}\t{split_line[0]}\t{split_line[1]}\t{split_line[2]}\t{split_line[3]}",
                              file=write)


def plot_kappa():
    main_directory = "stage_one/runs_at_each_lambda"
    expected_kappas = np.loadtxt("nvt_simulations/solid/T_400/analysis/kappa.txt")
    for directory in [f"{main_directory}/well_width_0.09", f"{main_directory}/well_width_0.9",
                      f"{main_directory}/well_width_4.0"]:
        data = np.loadtxt(f"{directory}/kappa.txt")
        plt.figure()
        for i, atom_type in enumerate(np.unique(data[:, 1])):
            mask = data[:, 1] == atom_type
            assert np.all(data[mask, 0] == np.array([i * 20000 / 2000000 for i in range(1, 2000000 // 20000 + 1)]))
            assert np.all(data[mask, 1] == atom_type)
            plt.errorbar(data[mask, 0], data[mask, 3], yerr=data[mask, 4], label=f"Type {int(atom_type)}",
                         color=f"C{i}")
            expected_mask = expected_kappas[:, 0] == atom_type
            assert np.all(expected_kappas[expected_mask, 0] == atom_type)
            assert len(expected_kappas[expected_mask, 2]) == 1
            plt.axhline(expected_kappas[expected_mask, 2][0], color=f"C{i}", linestyle="--")
            plt.axhspan(expected_kappas[expected_mask, 2][0] + expected_kappas[expected_mask, 3][0],
                        expected_kappas[expected_mask, 2][0] - expected_kappas[expected_mask, 3][0],
                        color=f"C{i}", alpha=0.2)
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\kappa$ (kcal/mol)")
        plt.legend()
        plt.savefig(f"{directory}/kappa.pdf")
        plt.close()


if __name__ == '__main__':
    #main()
    plot_kappa()
