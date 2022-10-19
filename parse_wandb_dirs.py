import os
import matplotlib as mpl
import pickle

import pandas.errors

from classes import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    """
    - args.dir
    -- parent_dir
    --- exp_dir
    --- exp_dir
    -- parent_dir
    --- exp_dir
    --- exp_dir
    """
    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)
        assert all([k in config.keys() for k in ['parse', 'experiment_dir', 'pickle_save_path']])
        exp_dir = config.get('experiment_dir')
        assert os.path.isdir(exp_dir), f"Path {exp_dir} is not a directory!"
        os.chdir(exp_dir)
        save_path = config.get('pickle_save_path')
        assert os.path.exists(base_path := os.path.split(save_path)[0]), \
            f"Base path {base_path} of pickle_save_path doesn't exist!"
    group = build_group_from_config(config.get('parse'))

    columns_seen = set()
    for exp_parent_dir in os.scandir(exp_dir):
        if os.path.isdir(exp_parent_dir):
            for exp_dir in os.scandir(exp_parent_dir):
                log_prefix = f"{exp_dir.path}: \n\t"
                wandb_run_files = None
                if os.path.exists(wandb_path := os.path.join(exp_dir, 'wandb')):
                    wandb_run_files = [fn for fn in os.listdir(wandb_path) if fn.startswith("run")]
                if wandb_run_files is None:
                    continue
                assert len(wandb_run_files) == 1
                with open(os.path.join(exp_dir, 'wandb', wandb_run_files[0], 'files', 'config.yaml'), 'r') as file:
                    config = yaml.safe_load(file)
                path_to_progress_csv = os.path.join(exp_dir, 'progress.csv')
                try:
                    data = pd.read_csv(path_to_progress_csv)
                    columns_seen |= set(data.columns)
                    config['data'] = data
                    err = group.add_run(config)
                    if err is not None:
                        print(log_prefix + err)
                except pandas.errors.EmptyDataError:
                    print(log_prefix + "progress.csv was empty. Skipping.")
                except KeyError as e:
                    print(log_prefix + f"Could not find keys {e.args[0]} in config.yaml. Skipping.")
    del data, config, exp_dir, file, wandb_run_files
    print("Columns seen: \n\t{}".format('\n\t'.join([c for c in columns_seen])))

    print(group)
    group.apply_operations()
    with open(os.path.join(save_path), 'wb') as f:
        pickle.dump(group, f, pickle.HIGHEST_PROTOCOL)
