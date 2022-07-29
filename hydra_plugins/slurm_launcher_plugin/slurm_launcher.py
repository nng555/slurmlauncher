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
from omegaconf import DictConfig, open_dict, MISSING, OmegaConf

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside launch()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.

log = logging.getLogger(__name__)

@dataclass
class SlurmConfig:
    _target_: str = (
        "hydra_plugins.slurm_launcher_plugin.slurm_launcher.SlurmLauncher"
    )

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
    max_running: int = -1
    max_pending: int = -1
    max_total: int = -1
    wait_time: int = 10
    env_type: str = 'venv'
    env_name: Optional[str] = None
    job_name: str = '${hydra.job.name}'
    job_dir: str = '${hydra.sweep.dir}'
    modules: Optional[str] = None
    time: Optional[str] = None

ConfigStore.instance().store(
    group="hydra/launcher", name="slurm", node=SlurmConfig,
)

class SlurmLauncher(Launcher):

    def __init__(self,
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
                 job_name: str,
                 job_dir: str,
                 modules: str,
                 time: str,
    ) -> None:
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        self.job_name = job_name
        self.job_dir = job_dir

        if modules is not None:
            self.modules = modules.split(',')
        else:
            self.modules = None

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
                'time': time,
        }

        self.max_running = max_running
        self.max_pending = max_pending
        self.max_total = max_total

        self.wait_time = wait_time

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

    def filter_overrides(self, overrides):
        """
        :param overrides: overrides list
        :return: returning a new overrides list with all the keys starting with hydra. filtered.
        """
        overrides = list(overrides)

        # escape characters for command line execution
        for i in range(len(overrides)):
            opt, val = overrides[i].split('=', 1)
            if "$" in val:
                val = val.replace('$', '\$')
            else:
                overrides[i] = '='.join([opt, '"' + val + '"'])

        return [x for x in overrides if not x.startswith("hydra.")]

    def launch_job(self, slrm_fname):
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

        subprocess.run(['sbatch', slrm_fname])

    def write_slurm(self, slrm_fname, overrides):
        # set up run directories
        curr_cwd = os.getcwd()
        exec_path = os.path.join(curr_cwd, sys.argv[0])

        if self.env_type == 'conda':
            venv_sh = 'source activate {}'.format(self.env_name)
        elif self.env_type == 'venv':
            venv_sh = '. $HOME/venv/{}/bin/activate'.format(self.env_name)
        else:
            venv_sh = ''
        slurm_opts = ['#SBATCH --' + k.replace('_','-') + '=' + v for k, v in self.slurm_kwargs.items() if v is not None]
        slurm_opts = ['#!/bin/bash'] + slurm_opts

        if self.modules is not None:
            ml_sh = ' '.join(['module load'] + self.modules) + '\n'
        else:
            ml_sh = ''

        # TODO: add checkpoint symlinking
        sh_str = """
{0}
{1}
python3 {2} {3}""".format(
                ml_sh,
                venv_sh,
                exec_path,
                overrides,
            )

        # write slurm file
        with open(slrm_fname, 'w') as slrmf:
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

        # tag with extra tags and idx to ensure no overlap
        tags = getattr(self.config, 'tags', None)
        has_tags = (tags is not None and len(tags) != 0)
        #TODO: assert tag is a list here
        if not has_tags:
            # load tags automatically from overrides
            multi_overrides = self.config.hydra.overrides
            tags = []
            for v in multi_overrides.values():
                for o in v:
                    key, vals = o.split('=')
                    # a bit janky but check if val is a list or dict
                    if (vals[0] == '[' and vals[-1] == ']') or \
                       (vals[0] == '{' and vals[-1] == '}'):
                           continue
                    elif ',' in vals:
                        tags.append(key.split('.')[-1] + '=${{{}}}'.format(key))

        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config, list(overrides)
            )

            # add tags to sweep_config for resolution
            if not has_tags:
                OmegaConf.update(sweep_config, 'tags', tags)

            tag = ','.join([str(t) for t in sweep_config.tags])

            # add manual override to launcher
            if not has_tags:
                overrides.append('tags=[{}]'.format(tag))

            import pdb; pdb.set_trace()

            job_dir = os.path.join(self.job_dir, tag)

            # set up run directories
            log_dir = os.path.join(job_dir, "log")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.slurm_kwargs['output'] = os.path.join(job_dir, 'log/%j.out')
            self.slurm_kwargs['error'] = os.path.join(job_dir, 'log/%j.err')
            self.slurm_kwargs['job_name'] = self.job_name + '/' + tag

            slrm_fname = os.path.join(job_dir, 'launch.slrm')
            overrides = self.filter_overrides(overrides)
            self.write_slurm(slrm_fname, " ".join(overrides))

            with open_dict(sweep_config):
                sweep_config.hydra.job.id = f"job_id_for_{idx}"
                sweep_config.hydra.job.num = idx
            HydraConfig.instance().set_config(sweep_config)

            lst = " ".join(overrides)
            log.info(f"\t#{idx} : {lst}")
            log.info("\tJob tag: {}".format(tag))

            self.launch_job(slrm_fname)

            configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        return runs
