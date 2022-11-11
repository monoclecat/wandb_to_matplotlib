# Create plots from WandB experiments
This tool can help you download, group, aggregate, and plot your WandB experiments across multiple folders in nested directories. 

The configuration keys you want to group the experiment metrics by must have been 
saved with WandB using `wandb.config.update(your_config)`

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
This tool follows an incremental approach. 
The yaml config file defines all behavior. 
An example yaml is provided in this directory. 

### 1. Download metrics
Populate the `root_dir`, `csv_file_name`, `pickle_save_path`, `wandb_entity` and `wandb_project` fields of your yaml. 
Then, run 
```bash
# Make sure you venv is activated and that you are in the correct directory!
python 01_download_metrics.py -c path-to-your-yaml
```
to download the metrics and save them in the experiment folders as csv files. 
This script will build a directory structure in the `root_dir` which is also built by wandb when running the experiment. 
```
experiment_dir
├── ... any depth of directories
│    ├── wandb_experiment1
│    │    ├── progress.csv  # Will be created by parse_wandb_dirs.py
│    │    └── wandb
│    │         └── run-...
│    │              └── files
│    │                   └── config.yaml
│    │
│   ... any number of experiment directories with the structure of wandb_experiment1
│    
... any number of group folders containing experiment directories
```
Existing `config.yaml` files will not be overwritten. 

### 2. Find plottable data and keys to group by
The script
```bash
python 02_explore_metrics.py -c path-to-your-yaml
```
will give you key paths you can group by and the names of metrics you can plot. 
You can now populate the `parse` field of your yaml with the hierarchical grouping that you want to group your 
metrics by before plotting. 
In the final `child_group`, put in the names of the metrics you want to plot. 
At the level of where you want to aggregate your metrics, add the appropriate `operations`. 

### 3. Group and aggregate metrics
The requested grouping and aggregation will be performed in 
```bash
python 03_explore_metrics.py -c path-to-your-yaml
```
It will save the result in a pickle file. 

### 4. Plotting
Now you can populate the `plot` field of your yaml. 
Running 
```bash
python 03_explore_metrics.py -c path-to-your-yaml
```
will plot the grouped and aggregated metrics that were pickled in the previous step. 
