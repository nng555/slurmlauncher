import os
import logging
import subprocess
import datetime
import sys
import time

from hydra import utils
from pathlib import Path
from omegaconf import OmegaConf, listconfig

log = logging.getLogger(__name__)

# username
USER = os.environ['USER']

# home directory
HOME = '/h/' + USER

# path to venv directory if using
VENV_DIR = HOME + '/venv/'

# slurm job folder
SLURM_DIR = '/checkpoint/' + USER

def filter_overrides(overrides):
    """
    :param overrides: overrides list
    :return: returning a new overrides list with all the keys starting with hydra. filtered.
    """
    overrides = list(overrides)

    # add dynamic evaluation of config values
    for i in range(len(overrides)):
        opt, val = overrides[i].split('=', 1)
        if "$" in val:
            val = val.replace('$', '\$')
        overrides[i] = '='.join([opt, '"' + val + '"'])

    return [x for x in overrides if not x.startswith("hydra.")]

def eval_val(val):
    if 'eval:' in str(val):
        return val.split('eval:', 1)[0] + str(eval(val.split('eval:', 1)[1]))
    else:
        return str(val)

def resolve_name(name):
    if isinstance(name, listconfig.ListConfig):
        name_list = []
        for i in range(len(name)):
            if name[i] is not None:
                if isinstance(name[i], listconfig.ListConfig):
                    name_list.append('_'.join(name[i]))
                else:
                    name_list.append(eval_val(str(name[i])))
        return '_'.join(name_list)
    else:
        return eval_val(name)

def get_j_dir(cfg):
    if hasattr(cfg.slurm, 'date') and cfg.slurm.date:
        date = cfg.slurm.date
    else:
        date = datetime.datetime.now()
        date = date.strftime("%Y-%m-%d")

    # set default slurm directory here. By default is set to $HOME/slurm/${date}/${job.name}
    return os.path.join(HOME, "slurm", date, resolve_name(cfg.slurm.job_name))

def write_slurm(cfg):

    # set up run directories
    j_dir = get_j_dir(cfg)

    scripts_dir = os.path.join(j_dir, "scripts")
    print(scripts_dir)
    if not os.path.exists(scripts_dir):
        Path(scripts_dir).mkdir(parents=True, exist_ok=True)

    slurm_opts = ['#SBATCH --' + k.replace('_','-') + '=' + resolve_name(v) for k, v in cfg.slurm.items() if (v != None and k != 'date')]

    # default output and error directories
    if 'output' not in cfg.slurm:
        slurm_opts.append('#SBATCH --output={}/log/%j.out'.format(j_dir))
    if 'error' not in cfg.slurm:
        slurm_opts.append('#SBATCH --error={}/log/%j.err'.format(j_dir))

    slurm_opts = ['#!/bin/bash'] + slurm_opts + ['bash {0}/scripts/{1}.sh'.format(j_dir, resolve_name(cfg.slurm.job_name))]

    # write slurm file
    with open(os.path.join(j_dir, "scripts", resolve_name(cfg.slurm.job_name) + '.slrm'), 'w') as slrmf:
        slrmf.write('\n'.join(slurm_opts))

def write_sh(cfg, overrides):

    if hasattr(cfg.slurm, 'date') and cfg.slurm.date:
        date = cfg.slurm.date
    else:
        date = datetime.datetime.now()
        date = date.strftime("%Y-%m-%d")

    # set up run directories
    j_dir = get_j_dir(cfg)
    hydra_cwd = os.getcwd()
    curr_cwd = utils.get_original_cwd()
    exec_path = os.path.join(curr_cwd, sys.argv[0])

    scripts_dir = os.path.join(j_dir, "scripts")
    if not os.path.exists(scripts_dir):
        Path(scripts_dir).mkdir(parents=True, exist_ok=True)

    if 'conda' in cfg:
        venv_sh = 'conda activate {}'.format(cfg.conda)
    elif 'venv' in cfg:
        venv_sh = '. {}/{}/bin/activate'.format(VENV_DIR, cfg.venv)
    else:
        venv_sh = ''

    with open(os.path.join(j_dir, "scripts", resolve_name(cfg.slurm.job_name) + '.sh'), 'w') as shf:
        shf.write(
"""#!/bin/bash
if [ ! -d {0}/$SLURM_JOB_ID ]; then
    ln -s {6}/$SLURM_JOB_ID {0}/$SLURM_JOB_ID
fi
touch {0}/$SLURM_JOB_ID/DELAYPURGE
{2}
python3 {3} {4} slurm.date="{5}"
""".format(
            j_dir,
            hydra_cwd,
            venv_sh,
            exec_path,
            overrides,
            date,
            SLURM_DIR,
        ))

def symlink_hydra(cfg, cwd):
    hydra_dir = os.path.join(get_j_dir(cfg), 'conf')
    if not os.path.exists(os.path.join(hydra_dir, os.environ['SLURM_JOB_ID'])):
        log.info('Symlinking {} : {}'.format(cwd, hydra_dir))
        if not os.path.exists(hydra_dir):
            Path(hydra_dir).mkdir(parents=True, exist_ok=True)
        os.symlink(cwd, os.path.join(hydra_dir, os.environ['SLURM_JOB_ID']), target_is_directory=True)

def launch_job(cfg):

    # set up run directories
    j_dir = get_j_dir(cfg)
    log_dir = os.path.join(j_dir, "log")
    if not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # launch safe only when < 100 jobs running
    while(True):
        num_running = int(subprocess.run('squeue -u $USER | grep R | wc -l', shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')) - 1
        num_pending = int(subprocess.run('squeue -u $USER | grep PD | wc -l', shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8'))
        num_total = num_running + num_pending

        if (cfg.max_running == -1 or num_running < cfg.max_running) and \
           (cfg.max_pending == -1 or num_pending < cfg.max_pending) and \
           (cfg.max_total == -1 or num_total < cfg.max_total):
               break
        print("{} jobs running and {} jobs pending, waiting...".format(num_running, num_pending))
        time.sleep(10)

    subprocess.run('sbatch {0}/scripts/{1}.slrm'.format(j_dir, resolve_name(cfg.slurm.job_name)), shell=True)
