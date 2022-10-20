import os
import pickle
from util import build_group_from_config, find_wandb_dirs, download_experiment_metrics
import pandas as pd
import pandas.errors
import yaml


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)
        assert all([k in config.keys() for k in ['parse', 'root_dir', 'pickle_save_path', 'csv_file_name']])
        root_dir = config.get('root_dir')
        assert os.path.isdir(root_dir), f"Path {root_dir} is not a directory!"
        os.chdir(root_dir)
        save_path = config.get('pickle_save_path')
        assert os.path.exists(base_path := os.path.split(save_path)[0]), \
            f"Base path {base_path} of pickle_save_path doesn't exist!"
    group = build_group_from_config(config.get('parse'))

    columns_seen = set()
    exp_dirs = find_wandb_dirs(root_dir)
    for exp_dir in exp_dirs:
        log_prefix = f"{exp_dir}: \n\t"
        wandb_run_dir = None
        if os.path.exists(wandb_path := os.path.join(exp_dir, 'wandb')):
            wandb_run_dir = [fn for fn in os.listdir(wandb_path) if fn.startswith("run")]
        if wandb_run_dir is None:
            continue
        assert len(wandb_run_dir) == 1
        wandb_run_dir = wandb_run_dir[0]
        with open(os.path.join(exp_dir, 'wandb', wandb_run_dir, 'files', 'config.yaml'), 'r') as file:
            exp_config = yaml.safe_load(file)
        path_to_progress_csv = os.path.join(exp_dir, config.get('csv_file_name'))
        if not os.path.exists(path_to_progress_csv):
            wandb_entity = config.get('wandb_entity')
            wandb_proj = config.get('wandb_project')
            assert wandb_entity is not None and wandb_proj is not None, \
                "In order to connect with the WandB API, the wandb_entity and wandb_project " \
                "must be provided in the yaml!"
            run_name_components = wandb_run_dir.split('-')
            run_api_identifier = f"{wandb_entity}/{wandb_proj}/{run_name_components[-1]}"
            print(log_prefix + "Downloading metrics from WandB")
            data = download_experiment_metrics(run_identifier=run_api_identifier)
            pd.DataFrame.from_dict(data).to_csv(path_to_progress_csv)
        try:
            data = pd.read_csv(path_to_progress_csv)
            columns_seen |= set(data.columns)
            exp_config['data'] = data
            err = group.add_run(exp_config)
            if err is not None:
                print(log_prefix + err)
        except pandas.errors.EmptyDataError:
            print(log_prefix + f"{config.get('csv_file_name')} was empty. Skipping.")
        except KeyError as e:
            print(log_prefix + f"Could not find keys {e.args[0]} in config.yaml. Skipping.")
    print("Columns seen: \n\t{}".format('\n\t'.join([c for c in columns_seen])))

    print(group)
    group.apply_operations()
    with open(os.path.join(save_path), 'wb') as f:
        pickle.dump(group, f, pickle.HIGHEST_PROTOCOL)
