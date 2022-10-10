import os
import matplotlib as mpl
import pickle

import pandas.errors

from classes import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, required=True,
                        help='Directory containing experiment directories (the project directory)')
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
        config = yaml.safe_load(file)
    group = build_group_from_kwargs(config)

    columns_seen = set()
    assert os.path.isdir(args.dir), f"Path {args.dir} is not a directory!"
    for exp_parent_dir in os.scandir(args.dir):
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
    with open(os.path.join(args.dir, f"{group.key}.pickle"), 'wb') as f:
        pickle.dump(group, f, pickle.HIGHEST_PROTOCOL)
