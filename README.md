# Slurm Launcher Plugin

Plugin files for launching jobs on a slurm cluster with Hydra. Currently works only with Hydra 1.0 to accomodate the use of fairseq as well, but fixing it to work with the latest dev version isn't too hard.

To install clone the repository (and ensure hydra is installed), then run
```
pip install -e .
```

This will install the slurm launcher plugin. To customize the launcher for your specific environment, you will need to change the global variables set in `slurm_utils.py`.
As a test, try running `python3 run_bash.py command="echo TEST" -m`, then checking the log files out `$HOME/slurm/${date}/run_bash/log/${job_id}.out`

# Using the Launcher

This slurm launcher acts as a layer of abstraction on top of a system for autogenerating `.sh` and `.slrm` files and launching slurm jobs. It presents a simple way to run many sets of repeatable experiments and organize the outputs with minimal overhead. The core of the plugin is the hydra configuration files.

## Setting up the Configuration
In order to use the slurm launcher, you should create a configuration folder that will contain the `.yaml` configuration files for your project as well as some base config files. An example base configuration is provided in `base_conf`. The basic structure of the folder is shown below.

```
.
├── config.yaml
└── slurm
    ├── cpu.yaml
    └── default.yaml
```

`config.yaml` stores the top level defaults for each group as well as configs for the launcher itself. Additional keys and defaults can be added here as necessary. Details on what each key is used for are provided below:
- `venv/conda`: name of virtual environment to run job with. If both are provided, `conda` takes priority
- `max_running`: max \# of running jobs. If limit is reached, launcher will wait until jobs have completed then continue launching
- `max_pending`: max \# of pendng jobs. If limit is reached, launcher will wait until jobs have launched then continue launching
- `max_total`: max \# of total jobs. If limit is reached, launcher will wait until jobs have launched or completed then continue launching

`slurm/` contains the configuration for the slurm job. These values can be easily tweaked or overriden to fit whatever slurm cluster you are running on. To add additional slurm options, simply add in the key value pair into this file (or another alternate configuration) and it will automatically be added to the `.slrm` file.

More details on setting up and using configuration files can be found [here](https://hydra.cc/docs/intro/)

## Dynamic Evaluation
This plugin adds in dynamic evaluation into configuration processing. Dynamic evluation is invoked using the keyword `eval:` at the start of a value. The launcher will then evaluate the expression after this keyword using python and interpolate the result as the value for that particular key.

As an example, in the `slurm/default.yaml` configuration file, we have the key/value pairs
```
cpus_per_task: eval:4*int("${slurm.gres}"[-1])
gres: gpu:1
```
Hydra will first interpolate `${slurm.gres}`, then evaluate the expression `4*int("gpu:1"[-1])` which returns `4`. This is then the value of `cpus_per_task` during runtime. 
Dynamic evaluation is useful for setting arguments based on the value from other arguments. In this case, we would like to request 4 times as many CPUs as there are GPUs.

## Decorating a Function
Modifying an existing script to run with the slurm launcher is simple. First add in the import statements
```
import hydra
from omegaconf import DictConfig
from hydra import slurm_utils
```
to the top of your script. The `__main__` function of the script should be as follows:
```
if __name__ == "__main__":
    my_method()
```
Finally, add the decorator and modify the method signature as follows:
```
@hydra.main(config_path='/path/to/base_conf', config_name='config')
def my_method(cfg: DictConfig):
    slurm_utils.symlink_hydra(cfg, os.getcwd())
    ...
```
All values in the configuration will then be available to your script from within the `cfg` object. As an example, the `run_bash.py` script launches a given bash command (stored within `cfg.command`) on the slurm cluster.
Adding in the final symlinking command is not necessary but is useful if you need to check certain log easily. Note that including this line means that jobs cannot run locally.

## Running a Job
To launch your script on the cluster, simply run
```
python3 my_script.py -m
```
and that's it! The plugin will automatically generate the required `.sh` and `.slrm` files then launch your job for you. 
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

More details on [overrides](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/) and [sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) can be found in the [hydra tutorial](https://hydra.cc/docs/tutorials/intro/)

### Job Names
Job names are generated automatically from the `slurm.job_name` field. The field is typically a list of values that are concatenated with `_` to generate the final job name, but can be set manually to a single value.
In order to change the job name, simply add or remove items in the list. By default, the first value in the list is simply the name of the script. 

As an example, if we are training a model and wanted to include the training data, model name, and learning rate, our `job_name` field might look something like
```
job_name:
  - eval:sys.argv[0][:-3].split('/')[-1]
  - ${data._name}
  - ${model._name}
  - ${train.lr}
```
If any of these values are null, they are simply omitted when generating the final job name.
