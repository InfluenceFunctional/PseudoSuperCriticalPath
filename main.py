import argparse

import numpy as np

from utils import process_config
from workflows.bulk_simulation import md_sampling
from workflows.free_energy_calculation import free_energy
from workflows.lambda_sampling import lambda_runs
from workflows.lambda_trajectory_generation import gen_run
from workflows.stage_two_re_box_calculation import re_box_energy_calc

'''
# TODO
legend: X -> to test on cluster
        x -> finished and tested
main:
x> config processing
-> calls to all execution modes

configs:
-> args to each module
-> combined workflows

modules:
init_MD: 
-> init structures via e2emolmats
X> various MD runs
stage_one:
X> gen run
X> lambda runs
x> free energy / stats
stage_two:
X> gen run
X> lambda runs
X> box reevaluations
-> free energy / stats
stage_three:
X> gen run
X> lambda runs
x> free energy / stats
overall:
-> DG calculation
'''

if __name__ == '__main__':
    '''
    parse arguments from config and command line and generate config namespace
    '''
    parser = argparse.ArgumentParser()
    _, cmdline_args = parser.parse_known_args()

    config = process_config(cmdline_args)

    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))

    '''
    run the code in selected mode
    '''
    if config.npt_simulation:
        temperatures_list = np.arange(config.reference_temperature - config.temperature_span,
                                      config.reference_temperature + config.temperature_span + 1,
                                      config.temperature_delta)
        md_sampling(
            'npt',
            config.structure_name,
            list(temperatures_list),
            config.runs_directory,
            config.sampling_time,
            config.equilibration_time
        )
    if config.nvt_simulation:
        md_sampling(
            'nvt',
            config.structure_name,
            [config.reference_temperature],
            config.runs_directory,
            config.sampling_time,
            None
        )

    if config.stage_one_gen:
        gen_run('stage_one',
                config.structure_name,
                config.reference_temperature,
                config.runs_directory,
                config.num_restarts,
                config.sampling_time)
    if config.stage_two_gen:
        gen_run('stage_two',
                config.structure_name,
                config.reference_temperature,
                config.runs_directory,
                config.num_restarts,
                config.sampling_time)
    if config.stage_three_gen:
        gen_run('stage_three',
                config.structure_name,
                config.reference_temperature,
                config.runs_directory,
                config.num_restarts,
                config.sampling_time)

    if config.stage_one_lambda:
        lambda_runs('stage_one',
                    config.structure_name,
                    config.reference_temperature,
                    config.runs_directory,
                    config.num_restarts,
                    config.sampling_time,
                    config.restart_sampling_time,
                    config.stage_one_sampling_dir
                    )
    if config.stage_two_lambda:
        lambda_runs('stage_two',
                    config.structure_name,
                    config.reference_temperature,
                    config.runs_directory,
                    config.num_restarts,
                    config.sampling_time,
                    config.restart_sampling_time,
                    config.stage_two_num_restarts
                    )
    if config.stage_three_lambda:
        lambda_runs('stage_three',
                    config.structure_name,
                    config.reference_temperature,
                    config.runs_directory,
                    config.num_restarts,
                    config.sampling_time,
                    config.restart_sampling_time,
                    )

    if config.stage_two_re_box:
        re_box_energy_calc(config.structure_name,
                           config.reference_temperature,
                           config.runs_directory)

    if config.stage_one_free_energy:
        free_energy('stage_one',
                    config.structure_name,
                    config.reference_temperature,
                    config.runs_directory,
                    plot_trajs=True)
    if config.stage_two_free_energy:
        free_energy('stage_two',
                    config.structure_name,
                    config.reference_temperature,
                    config.runs_directory,
                    plot_trajs=True
                    )
    if config.stage_three_free_energy:
        free_energy('stage_three',
                    config.structure_name,
                    config.reference_temperature,
                    config.runs_directory,
                    plot_trajs=True)
