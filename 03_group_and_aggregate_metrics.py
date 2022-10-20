import os
import pickle
from util import find_wandb_dirs, populate_group, build_group_from_config, parse_config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = parse_config(args.config)
    exp_dirs = find_wandb_dirs(config.get('root_dir'))

    group = build_group_from_config(config.get('parse'))
    populate_group(exp_dirs, config, group)
    print(group)
    group.apply_operations()
    with open(os.path.join(config.get('pickle_save_path')), 'wb') as f:
        pickle.dump(group, f, pickle.HIGHEST_PROTOCOL)
