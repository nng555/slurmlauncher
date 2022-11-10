from dataclasses import dataclass
from enum import Enum

import sys
import subprocess
import os
import logging
import datetime
from pathlib import Path
from typing import Optional, Sequence
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
from omegaconf import DictConfig, open_dict, MISSING, OmegaConf, ListConfig

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside launch()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.

log = logging.getLogger(__name__)

class Cluster(Enum):
    SLURM = 0
    LSF = 1

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
    partition: Optional[str] = None
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
    date: str = datetime.datetime.now().strftime('%Y-%m-%d')
    override_tags: bool = False
    sort_tags: bool = True
    cluster: Cluster = Cluster.SLURM

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
                 date: str,
                 override_tags: bool,
                 sort_tags: bool,
                 cluster: Cluster,
    ) -> None:
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        self.cluster = cluster
        if partition is None and self.cluster == Cluster.SLURM:
            raise Exception("Partition required for slurm clusters")

        self.job_name = job_name
        self.job_dir = job_dir

        self.date = date

        if modules is not None:
            self.modules = modules.split(',')
        else:
            self.modules = None

        if self.cluster == Cluster.SLURM:
            assert partition is not None, "Must provide partition when using SLURM"
            assert cpus_per_task is not None, "Must provide cpus_per_task when using SLURM"
            self.batch_kwargs = [
                ('cpus_per_task', str(cpus_per_task)),
                ('exclude', exclude),
                ('gres', gres),
                ('mem', mem),
                ('nodes', str(nodes)),
                ('ntasks_per_node', str(ntasks_per_node)),
                ('open_mode', open_mode),
                ('partition', partition),
                ('account', account),
                ('qos', qos),
                ('time', time),
            ]
        else:
            ex_nodes = ' && '.join(['hname!=' + node for node in exclude.split(',')])
            self.batch_kwargs = [
                ('R', f'"rusage[mem={mem}:duration=24h]"'),
                ('R', f'"span[ptile={cpus_per_task}] {ex_nodes}"'),
                # exclusive process by default so we don't clash with other jobs
                ('gpu', f'"num={gres[-1]}:mode=exclusive_process"'),
                ('n', str(ntasks_per_node)),
                ('q', qos),
                #'nnodes': str(nodes),
            ]

        self.max_running = max_running
        self.max_pending = max_pending
        self.max_total = max_total

        self.wait_time = wait_time

        self.env_type = env_type
        self.env_name = env_name

        self.override_tags = override_tags
        self.sort_tags = sort_tags

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

        # don't remove hydra overrides?
        return overrides

    def launch_job(self, batch_fname):
        # launch safe only when < 100 jobs running
        while(True):
            if self.cluster == Cluster.SLURM:
                num_running = int(subprocess.run('squeue -u $USER | grep R | wc -l',
                    shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')) - 1
                num_pending = int(subprocess.run('squeue -u $USER | grep PD | wc -l',
                    shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8'))
            else:
                num_running = int(subprocess.run('bjobs -u $USER | grep RUN | wc -l',
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8')) - 1
                num_pending = int(subprocess.run('bjobs -u $USER | grep PEND | wc -l',
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8'))
            num_total = num_running + num_pending

            if (self.max_running == -1 or num_running < self.max_running) and \
               (self.max_pending == -1 or num_pending < self.max_pending) and \
               (self.max_total == -1 or num_total < self.max_total):
                   break
            log.info("{} jobs running and {} jobs pending, waiting {} seconds...".format(num_running, num_pending, self.wait_time))
            time.sleep(self.wait_time)

        if self.cluster == Cluster.SLURM:
            subprocess.run(['sbatch', batch_fname])
        elif self.cluster == Cluster.LSF:
            with open(batch_fname, 'r') as infile:
                subprocess.run(['bsub'], stdin=infile)

    def write_batch(self, batch_fname, overrides, add_kwargs, job_name):
        # set up run directories
        curr_cwd = os.getcwd()
        exec_path = os.path.join(curr_cwd, sys.argv[0])

        run_lines = ['#!/bin/bash\n']

        # job options
        if self.cluster == Cluster.SLURM:
            opt_key = '#SBATCH --'
        elif self.cluster == Cluster.LSF:
            opt_key = '#BSUB -'
        batch_opts =  [opt_key + k.replace('_','-') + ' ' + v for (k, v) in (self.batch_kwargs + add_kwargs) if v is not None]
        run_lines.append('\n'.join(batch_opts) + '\n')

        # load modules
        if self.modules is not None:
            ml_sh = ' '.join(['module load'] + self.modules) + '\n'
        else:
            ml_sh = ''
        run_lines.append(ml_sh)

        # load environment
        if self.env_name is not None:
            if self.env_type == 'conda':
                venv_sh = 'conda deactivate\nsource activate {}\n'.format(self.env_name)
            elif self.env_type == 'venv':
                venv_sh = '. $HOME/venv/{}/bin/activate\n'.format(self.env_name)
            else:
                venv_sh = ''
            run_lines.append(venv_sh)

        # set env variables
        if self.cluster == Cluster.SLURM:
            job_id_name = 'SLURM_JOB_ID'
        elif self.cluster == Cluster.LSF:
            job_id_name = 'LSB_JOBID'
        env_sh = 'export JOBID=${}\n'.format(job_id_name)
        env_sh += 'export WANDB_NAME={}\n'.format(job_name)
        run_lines.append(env_sh)

        # run script
        run_lines.append('python3 {} {}\n'.format(exec_path, overrides))

        # write batch submission file
        with open(batch_fname, 'w') as batchf:
            batchf.writelines(run_lines)

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
        log.info("Launching {} jobs on {}".format(len(job_overrides), self.cluster.name))
        log.info("Job name: {}".format(self.job_name))
        runs = []

        # tag with extra tags and idx to ensure no overlap
        tags = getattr(self.config, 'tags', [])
        if tags is None:
            tags = []
        if isinstance(tags, str):
            tags = [tags]
        assert isinstance(tags, list) or isinstance(tags, ListConfig), \
                'tags must be a string or list if specified'

        sweep_keys = []
        # get keys we're sweeping over if not present or if not overridden
        if not self.override_tags:
            # get keys automatically from overrides
            multi_overrides = self.config.hydra.overrides
            for v in multi_overrides.values():
                for o in v:
                    key, vals = o.split('=')
                    # a bit janky but check if val is a list or dict or string
                    if (vals[0] == '[' and vals[-1] == ']') or \
                       (vals[0] == '{' and vals[-1] == '}') or \
                       (vals[0] == '"' and vals[-1] == '"') or \
                       (vals[0] == "'" and vals[-1] == "'"):
                           continue
                    elif ',' in vals:
                        assert vals != 'tags', 'tags cannot be swept over'
                        sweep_keys.append(key)

        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config, list(overrides)
            )
            choices = sweep_config.hydra.runtime.choices

            sweep_tags = tags.copy()

            # if not override then autopopulate from values
            if not self.override_tags:
                for job_or in sweep_config.hydra.overrides.task:
                    okey, val = job_or.split('=')
                    # grab last bit of keys swept over
                    if okey in sweep_keys:
                        if '.' in okey:
                            okey = okey.split('.')[-1]
                        sweep_tags.append(okey + '_' + val)
                    # grab last bit of defaults that are manually set
                    elif okey in choices:
                        if '/' in okey:
                            okey = okey.split('/')[-1]
                        sweep_tags.append(okey + '_' + choices[okey])

            if self.sort_tags:
                sweep_tags.sort()

            if len(sweep_tags) > 0:
                OmegaConf.update(sweep_config, 'tags', sweep_tags)
                tag = ','.join([str(t) for t in sweep_config.tags])
            else:
                tag = str(idx)

            # add manual override to launcher
            overrides.append('tags=[{}]'.format(tag))

            # manual override date in case we launch at a different day than we schedule
            overrides.append('hydra.launcher.date={}'.format(self.date))

            job_dir = os.path.join(self.job_dir, tag)

            # set up run directories
            log_dir = os.path.join(job_dir, "log")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            add_kwargs = []
            if self.cluster == Cluster.SLURM:
                add_kwargs.append(('output', os.path.join(job_dir, 'log/%j.out')))
                add_kwargs.append(('error', os.path.join(job_dir, 'log/%j.err')))
                add_kwargs.append(('job_name', self.job_name + '/' + tag))
                add_kwargs.append(('chdir', job_dir))
            else:
                add_kwargs.append(('o', os.path.join(job_dir, 'log/%J.out')))
                add_kwargs.append(('e', os.path.join(job_dir, 'log/%J.err')))
                add_kwargs.append(('J', self.job_name + '/' + tag))
                add_kwargs.append(('cwd', job_dir))

            batch_fname = os.path.join(job_dir, 'launch.sh')
            overrides = self.filter_overrides(overrides)
            self.write_batch(batch_fname, " ".join(overrides), add_kwargs, tag)

            with open_dict(sweep_config):
                sweep_config.hydra.job.id = f"job_id_for_{idx}"
                sweep_config.hydra.job.num = idx
            HydraConfig.instance().set_config(sweep_config)

            lst = " ".join(overrides)
            log.info(f"\t#{idx} : {tag}")
            #log.info("\tJob tag: {}".format(tag))
            self.launch_job(batch_fname)

            configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        return runs
