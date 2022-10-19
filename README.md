# Create plots from scattered WandB experiments
This tool can help you group, aggregate and plot your WandB experiments across multiple folders in nested directories. 

## Prerequisites 
Certain prerequisites need to be met to use this tool. 
1. In the experiments, the configuration keys to group by must have been saved with WandB using `wandb.config.update(your_config)`
2. The metrics you need must have been logged separately in a csv called `progress.csv`. 
It may be possible to retrieve the metrics somehow else but this isn't implemented for now. 
3. The directories to group and aggregate over must have the following structure:
```
experiment_dir
├── wandb_group1
│    ├── wandb_experiment1_of_group1
│    │    ├── progress.csv  # The metrics need to be saved here
│    │    └── wandb
│    │         └── run-...
│    │              └── files
│    │                   └── config.yaml  # This is where wandb.config.update() saves its configuration 
│    │
│   ... any number of experiment directories with this structure
│    │
... any number of group folders containing experiment directories
```

## Installation
```bash
cd path-to-where-you-want-to-install-this
git clone git@github.com:monoclecat/wandb_to_matplotlib_tikz.git
cd wandb_to_matplotlib_tikz
python3.8 -m venv .  # Create a new virtual environment
source bin/activate
pip install -r requirements.txt
```

## Usage
### Grouping and aggregation
`parse_wandb_dirs.py` will group and aggregate logged metrics of your WandB experiments that are saved locally. 
The logged metrics must be present in the experiment directory as a csv named `progress.csv`. 
Grouping is done based on the WandB config you saved during runtime using `wandb.config.update(your_config)`.
For aggregation, you can select different NumPy functions such as min, max or mean. 
The aggregated metrics are pickled and saved. 

### Plotting
`plot_parsed_wandb_dirs.py` takes the pickled metrics and creates plots based on a yaml config specified by you 
and creates plots from them.

### Example
An example yaml is provided in this repository. 
It is tailored to the case I needed it for, but it might give you an idea on how to use it. 
Once you have your own config set up, you can run 
```bash
# Make sure you venv is activated and that you are in the correct directory!
python parse_wandb_dirs.py -c path_to_your_yaml
```
to parse the metrics and 
```bash
python plot_parsed_wandb_dirs.py -c path_to_your_yaml
```
to plot them.
