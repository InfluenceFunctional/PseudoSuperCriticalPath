import subprocess


def main():
    subprocess.run("python analyze.py bulk_fluid/0/screen.log bulk_fluid/0/tmp.out bulk_fluid/analysis "
                   "--ramp_time=2000000 --minimization", check=True, shell=True)
    subprocess.run("python analyze.py bulk_solid/0/screen.log bulk_fluid/0/tmp.out bulk_solid/analysis "
                   "--ramp_time=2000000 --minimization", check=True, shell=True)
    subprocess.run("python analyze_rdf.py bulk_fluid/0/traj.dump bulk_fluid/0/new_system.data bulk_fluid/analysis "
                   "--ramp_steps=2000000 --dump_interval=10000", check=True, shell=True)
    subprocess.run("python analyze_rdf.py bulk_solid/0/traj.dump bulk_solid/0/new_system.data bulk_solid/analysis "
                   "--ramp_steps=2000000 --dump_interval=10000", check=True, shell=True)


if __name__ == '__main__':
    main()
