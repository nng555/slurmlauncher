# Slurm Launcher Plugin

Slurm Launcher is a lightweight plugin for launching jobs on a slurm cluster with Hydra. There is a lot less machinery compared to a full framework like [submitit](https://github.com/facebookincubator/submitit), but this allows for greater control and customizability.

To install, clone the repository then run:
```
pip install -e .
```

# Using the Launcher

This slurm launcher acts as a layer of abstraction on top of a system for autogenerating `.slrm` files and launching slurm jobs as well as specifying subdirectories. It presents a simple way to run many sets of repeatable experiments and organize the outputs with minimal overhead. The core of the plugin is the hydra configuration file.

## Quickstart

If you simply want to launch an existing python script to a SLURM cluster using this plugin, follow these steps. An example script and configuration can be found in the `example/` folder.

1. Copy `config.yaml` to the same directory as your file.
2. Modify the `gres`, `mem`, and `partition`, as well as add any other `#SBATCH` parameters you want in `config.yaml` to work with your SLURM cluster.
3. Add the following import statements to the top of your script:
```
import hydra
from omegaconf import DictConfig
```
4. Ensure your `__main__` function of the script calls the method to be run, as below:
```
if __name__ == "__main__":
    my_method()
```
5. Add the following decorator and modify the method signature:
```
@hydra.main(config_path='.', config_name='config')
def my_method(cfg: DictConfig):
    ...
```
If your method takes in arguments, these must be moved into `config.yaml` and can be specified manually on the command line by overriding them using hydra.
More details on [overrides](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/) and [sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) can be found in the [hydra tutorial](https://hydra.cc/docs/tutorials/intro/).
6. Launch your script:
```
python my_app.py -m
```

## Setting up the Configuration

To use the launcher locally, the minimum requirements are to override the default launcher as shown below:

```
defaults:
  - override hydra/launcher: slurm
```

To run a job on the SLURM cluster, set the appropriate `sbatch` options in `hydra/launcher`. The only required options to be set are `gres`, `mem`, and `partition`. As an example from the config in the app in `example/`:

```
hydra:
  launcher:
    mem: 25G
    partition: gpu
    gres: gpu:1
```

A full list of options and descriptions can be found in the `SlurmConfig` dataclass in `hydra_plugins/slurm_launcher_plugin/slurm_launcher.py`. 
Most of these are the same as the options provided in a typical slurm script like `cpus_per_task` or `qos`, for which documentation can be found [here](https://slurm.schedmd.com/pdfs/summary.pdf).
In addition, the launcher plugin provides a few extra configurable options, described below:
- `max_running`: Maximum number of running jobs allowed at one time. If exceeded, launcher will wait until existing jobs have finished before scheduling more jobs. Default set to `-1` for no limit.
- `max_pending`: Same as `max_running` but for pending jobs. Default set to `-1` for no limit.
- `max_total`: Same as `max_running` and `max_pending` but for total number of jobs running and pending. Default set to `-1` for no limit.
- `wait_time`: Amount of time in seconds to wait before checking the number of running and pending jobs for resuming scheduling jobs. Default set to 10 seconds.
- `env_type`: Virtual environment type to load prior to executing script. Currently only supports `venv` and `conda`
- `env_name`: Name of virtual environment to load.
- `modules`: Modules to load into environment prior to executing script. Format as comma separated string.

Although not required, it is highly suggested to modify the default hydra run and sweep directories to organize outputs more effectively. 
A suggested configuration file is shown below, which organizes outputs in a folder called `slurm` in your home directory. 
Outputs are then organized by date, job name, and subdirectory, which is specified using the special `tags` configuration variable.
More details are provided in the `tags` section below.

```
hydra:
  sweep: 
    # subdir is built dynamically using the tags
    dir: ${oc.env:HOME}/slurm/${now:%Y-%m-%d}/${hydra.job.name}
  run:
    # once we're running set tags from conf
    dir: ${hydra.sweep.dir}/${join:${oc.select:tags,[]}}
  job_logging:
    handlers:
      file:
        filename: log/${oc.env:SLURM_JOB_ID,local}.log
```

## `tags` Configuration
Most SLURM launching plugins simply index jobs in a sweep by a number, which is unhelpful when searching for specific set of hyperparameters. 
The other option provided by hydra is to label subdirectories by all overrides, which can quickly become unwieldy when an application becomes complex.
As a middle point, this plugin uses a special `tags` variable that is dynamically loaded per job and specifies the hyperparameters of interest that are being swept across.
If no `tags` are specified, the launcher automatically fills the `tags` with the key name and value of the hyperparameters with multiple values.

If overridden, `tags` should be a set as a list of values or variables that can be used to identify a particular job in a sweep. This is useful if variable names need to be condensed. 
> :warning: Note that underspecifying the tags manually will lead to collisions in job names and potential unexpected behavior. 

For example, if we are sweeping across learning rate and batch size for a training job, we might run:
```
python3 train.py model=bert data=sts seed=0 train.lr=0.001,0.0001 train.batch_size=4,8,16 tags=[lr\${train.lr},bs\${train.batch_size}] -m
```
This indexes jobs using the learning rate and batch size. Alternatively, with no `tags` set, the `tags` will be autogenerated as `tags=[lr=\${train.lr},batch_size=\${train.batch_size}]`

## Job Names
Job names are generated automatically from the `hydra.job.name` variable and are set by default to the name of the script. For a more descriptive job name for identification on slurm and in the filesystem, you can override the value in your configuration.

## Running a Job
To launch your script on the cluster, simply run
```
python3 my_app.py -m
```
and that's it! The plugin will automatically generate the required `.slrm` files then launch your job for you. 
Values can be overriden as in hydra simply by specifying a particular group setting or specific value.
For example, if I wanted to run my script, but using a different dataset and with 4 GPUs instead of 1, I would run:
```
python3 my_script.py data=my_data slurm.gres=gpu:4 -m
```
To perform a hyperparameter sweep, simply set multiple values for each override and hydra will run a separate job for each configuration in the cartesian product.
```
python3 my_script.py train.lr=1e-3,1e-4,1e-5 train.optimizer=sgd,adam -m
```
Jobs can also be run locally by omitting the `-m` option. For example
```
python3 run_bash.py command="echo TEST"
```
will simply run the provided command locally rather than on the cluster.


