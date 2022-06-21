# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass

import sys
import subprocess
import os
import logging
import datetime
from pathlib import Path
from typing import Optional, Sequence
from hydra.utils import get_original_cwd
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)

from hydra.core.hydra_config import HydraConfig
from hydra.types import HydraContext
from hydra.core.config_store import ConfigStore
from hydra.core.singleton import Singleton
from hydra.plugins.launcher import Launcher
from hydra.types import TaskFunction
from omegaconf import DictConfig, open_dict, MISSING

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside launch()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.


log = logging.getLogger(__name__)


@dataclass
class LauncherConfig:
    _target_: str = (
        "hydra_plugins.slurm_launcher_plugin.slurm_launcher.SlurmLauncher"
    )
    date: Optional[str] = None
    cpus_per_task: Optional[int] = None
    exclude: Optional[str] = None
    gres: str = MISSING
    mem: str = MISSING
    nodes: int = 1
    ntasks_per_node: int = 1
    open_mode: str = 'append'
    partition: str = MISSING
    account: Optional[str] = None
    qos: Optional[str] = None
    max_running: int = 350
    max_pending: int = 350
    max_total: int = 400
    wait_time: int = 10
    env_type: str = 'venv'
    env_name: Optional[str] = None

ConfigStore.instance().store(
    group="hydra/launcher", name="slurm", node=LauncherConfig,
)

class SlurmLauncher(Launcher):

    def __init__(self,
                 date: str,
                 cpus_per_task: int,
                 exclude: str,
                 gres: str,
                 mem: int,
                 nodes: int,
                 ntasks_per_node: int,
                 open_mode: str,
                 partition: str,
                 account: str,
                 qos: str,
                 max_running: int,
                 max_pending: int,
                 max_total: int,
                 wait_time: int,
                 env_type: str,
                 env_name: str,
    ) -> None:
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        # foo and var are coming from the the plugin's configuration
        if date is not None:
            self.date = date
        else:
            date = datetime.datetime.now()
            self.date = date.strftime("%Y-%m-%d")

        self.job_name = 'test'

        self.slurm_kwargs = {
                'cpus_per_task': str(cpus_per_task),
                'exclude': exclude,
                'gres': gres,
                'mem': mem,
                'nodes': str(nodes),
                'ntasks_per_node': str(ntasks_per_node),
                'open_mode': open_mode,
                'partition': partition,
                'account': account,
                'qos': qos,
                'output': '{}/log/%j.out'.format(self.job_dir),
                'error': '{}/log/%J.err'.format(self.job_dir),
        }

        self.max_running = max_running
        self.max_pending = max_pending
        self.max_total = max_total

        self.wait_time = wait_time

        # set up run directories
        self.scripts_dir = os.path.join(self.job_dir, "scripts")
        if not os.path.exists(self.scripts_dir):
            Path(self.scripts_dir).mkdir(parents=True, exist_ok=True)
        self.slrm_fname = os.path.join(self.scripts_dir, self.job_name + '.slrm')

        # set up run directories
        self.log_dir = os.path.join(self.job_dir, "log")
        if not os.path.exists(self.log_dir):
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.env_type = env_type
        self.env_name = env_name

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    """
    def symlink_hydra(cfg, cwd):
        if 'SLURM_JOB_ID' in os.environ:
            hydra_dir = os.path.join(get_j_dir(cfg), 'conf')
            if not os.path.exists(os.path.join(hydra_dir, os.environ['SLURM_JOB_ID'])):
                log.info('Symlinking {} : {}'.format(cwd, hydra_dir))
                if not os.path.exists(hydra_dir):
                    Path(hydra_dir).mkdir(parents=True, exist_ok=True)
                os.symlink(cwd, os.path.join(hydra_dir, os.environ['SLURM_JOB_ID']), target_is_directory=True)
    """

    @property
    def job_dir(self):
        return os.path.join(os.environ['HOME'], "slurm", self.date, self.job_name)

    def filter_overrides(self, overrides):
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
            if opt == "command":
                overrides[i] = '='.join([opt, '\\"' + val + '\\"'])
            else:
                overrides[i] = '='.join([opt, '"' + val + '"'])

        return [x for x in overrides if not x.startswith("hydra.")]

    def eval_val(self, val):
        if 'eval:' in str(val):
            return val.split('eval:', 1)[0] + str(eval(val.split('eval:', 1)[1]))
        else:
            return str(val)

    def resolve_name(self, name):
        if isinstance(name, listconfig.ListConfig):
            name_list = []
            for i in range(len(name)):
                if name[i] is not None:
                    if isinstance(name[i], listconfig.ListConfig):
                        name_list.append('_'.join(name[i]))
                    else:
                        name_list.append(self.eval_val(str(name[i])))
            return '_'.join(name_list)
        else:
            return self.eval_val(name)

    def launch_job(self):
        # launch safe only when < 100 jobs running
        while(True):
            num_running = int(subprocess.run('squeue -u $USER | grep R | wc -l',
                shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')) - 1
            num_pending = int(subprocess.run('squeue -u $USER | grep PD | wc -l',
                shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8'))
            num_total = num_running + num_pending

            if (self.max_running == -1 or num_running < self.max_running) and \
               (self.max_pending == -1 or num_pending < self.max_pending) and \
               (self.max_total == -1 or num_total < self.max_total):
                   break
            print("{} jobs running and {} jobs pending, waiting...".format(num_running, num_pending))
            time.sleep(self.wait_time)

        subprocess.run(['sbatch', self.slrm_fname])

    def write_slurm(self, overrides):
        # set up run directories
        hydra_cwd = os.getcwd()
        curr_cwd = get_original_cwd()
        exec_path = os.path.join(curr_cwd, sys.argv[0])

        if self.env_type == 'conda':
            venv_sh = 'conda activate {}'.format(self.env_name)
        elif self.env_type == 'venv':
            venv_sh = '. $HOME/venv/{}/bin/activate'.format(self.env_name)
        else:
            venv_sh = ''

        slurm_opts = ['#SBATCH --' + k.replace('_','-') + '=' + v for k, v in self.slurm_kwargs.items() if v is not None]
        slurm_opts = ['#!/bin/bash'] + slurm_opts

        # TODO: add checkpoint symlinking
        sh_str = """
{0}
python3 {1} {2}""".format(
                venv_sh,
                exec_path,
                overrides,
            )

        # write slurm file
        with open(self.slrm_fname, 'w') as slrmf:
            slrmf.write('\n'.join(slurm_opts) + '\n')
            slrmf.write(sh_str)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> "JobReturn":
        setup_globals()
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = self.config.hydra.sweep.dir
        Path(str(sweep_dir)).mkdir(parents=True, exist_ok=True)
        log.info("Launching {} jobs on slurm".format(len(job_overrides)))
        runs = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(self.filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config, list(overrides)
            )

            with open_dict(sweep_config):
                sweep_config.hydra.job.id = f"job_id_for_{idx}"
                sweep_config.hydra.job.num = idx
            HydraConfig.instance().set_config(sweep_config)
            log.info("\tJob name : {}".format(self.job_name))

            self.write_slurm(" ".join(self.filter_overrides(overrides)))
            self.launch_job()

            configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        return runs
