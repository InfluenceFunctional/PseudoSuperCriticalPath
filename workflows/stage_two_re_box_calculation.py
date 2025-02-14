import subprocess
from pathlib import Path
import os
from random import shuffle
from shutil import copy

import numpy as np

template = '''
variable        xv equal v_a
variable        xyv equal v_b*cos(v_gamma*PI/180)
variable        yv equal sqrt(v_b*v_b-v_xyv*v_xyv)
variable        xzv equal v_c*cos(v_beta*PI/180)
variable        yzv equal (v_b*v_c*cos(v_alpha*PI/180)-v_xyv*v_xzv)/v_yv
variable        zv equal sqrt(v_c*v_c-v_xzv*v_xzv-v_yzv*v_yzv)

units           real
special_bonds   lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333
#neighbor        2.0 bin
#neigh_modify    delay 0 every 1 check yes page 1000000 one 20000 
atom_style      full
pair_style	 hybrid/overlay lj/cut 12.0 coul/long 12.0 

bond_style	 harmonic
dihedral_style	 charmm
angle_style	 harmonic
improper_style	 cvff

read_restart    restart.out.2.$variavel7

change_box_mol  all x final 0.0 ${a} y final 0.0 ${b} z final 0.0 ${c} xy final ${xyv} xz final ${xzv} yz final ${yzv} remap units box


'''


def re_box_energy_calc(structure_name: str,
                       reference_temperature: int,
                       runs_directory: Path,
                       ):
    run_type = 'stage_two'

    structure_directory = Path(runs_directory).joinpath(structure_name)
    if not os.path.exists(structure_directory):
        assert False, "Structure directory does not exist - missing bulk runs"

    stage_directory = structure_directory.joinpath(run_type)
    if not os.path.exists(stage_directory):
        assert False, "Missing lambda generation trajectory directory!"

    restarts_directory = stage_directory.joinpath('restart_runs')
    if not os.path.exists(restarts_directory):
        assert False, "Missing lambda restart runs directory!"

    run_md_name = run_type + '_box_change_' + 'run_MD.lmp'
    run_md_path = Path(__file__).parent.resolve().joinpath(run_md_name)

    lambda_runs = os.listdir(restarts_directory)
    lambda_runs = [restarts_directory.joinpath(Path(run)) for run in lambda_runs if 'lambda_' in run]

    lambda_inds = np.arange(len(lambda_runs))
    shuffle(lambda_inds)
    lambda_runs = [lambda_runs[ind] for ind in lambda_inds]
    # extract box information for each run
    box_params_dict = extract_stage_two_box_params(lambda_runs)

    for lambda_ind, run_dir in zip(lambda_inds, lambda_runs):
        '''go into the restart directory'''
        os.chdir(run_dir)

        outputs_path = 're_box_outputs'
        if not os.path.exists(outputs_path + '.npy'):
            np.save(outputs_path, {})

        restart_files = os.listdir()
        restart_files = [file for file in restart_files if 'stage_two_lambda_sample.restart' in file]
        restart_inds = np.arange(len(restart_files))
        shuffle(restart_inds)
        restart_files = [restart_files[ind] for ind in restart_inds]
        for restart_ind, restart_file in zip(restart_inds, restart_files):
            for lambda_ind2, run_dir2 in zip(lambda_inds, lambda_runs):
                run_index = f"{lambda_ind}_{restart_ind}_{lambda_ind2}"
                #logfile_name = run_index + ".log"
                output_dict = np.load(outputs_path + '.npy', allow_pickle=True).item()

                #if not os.path.exists(logfile_name):
                if run_index not in list(output_dict.keys()):
                    dx, dy, dz, xy, xz, yz = box_params_dict[str(run_dir2)]

                    with (open(run_md_path, "r") as read,
                          open(run_dir.joinpath(Path('box_change_run_MD.lmp')), "w") as write):
                        text = read.read()
                        text = text.replace("_T_SAMPLE", str(reference_temperature))
                        text = text.replace("_RESTART_FILE", restart_file)
                        text = text.replace('_XV', str(dx))
                        text = text.replace('_YV', str(dy))
                        text = text.replace('_ZV', str(dz))
                        text = text.replace('_XYV', str(xy))
                        text = text.replace('_XZV', str(xz))
                        text = text.replace('_YZV', str(yz))
                        #text = text.replace("_RUN_INDEX", run_index)
                        text = text.replace("_RUN_INDEX", str(0))
                        # always output to the same file, we're capturing the output directly anyway

                        write.write(text)

                    output = subprocess.run(['lmp', '-in', 'box_change_run_MD.lmp'], capture_output=True)
                    log = str(output.stdout)
                    traj = log.split('Performance:')[0].split('\\n')[-6:]
                    output_dict = np.load(outputs_path + '.npy', allow_pickle=True).item()
                    output_dict[run_index] = traj
                    np.save(outputs_path, output_dict)


def extract_stage_two_box_params(lambda_runs):
    box_params_dict = {}
    for lambda_ind, run_dir in enumerate(lambda_runs):
        os.chdir(run_dir)
        with open('stage_two_lambda_sample.data', 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'xlo xhi' in line:
                    elems = line.split(' ')
                    dx = float(elems[1]) - float(elems[0])
                elif 'ylo yhi' in line:
                    elems = line.split(' ')
                    dy = float(elems[1]) - float(elems[0])
                elif 'zlo zhi' in line:
                    elems = line.split(' ')
                    dz = float(elems[1]) - float(elems[0])
                elif 'xy xz yz' in line:
                    elems = line.split(' ')
                    xy, xz, yz = float(elems[0]), float(elems[1]), float(elems[2])

            box_params_dict[str(run_dir)] = dx, dy, dz, xy, xz, yz
    return box_params_dict
