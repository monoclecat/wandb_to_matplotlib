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
    # cmap = mpl.cm.get_cmap('tab10')
    group = build_group_from_kwargs(config)
    # group = Group(
            # key_path=['joint_failure_prob'],
            # readable_name='Joint Failure Prob.',
            # child_group=Group(
                # key_path=['policy_kwargs', 'actor_cspn_args', 'entropy_objective'],
                # readable_name='Entropy Objective',
                # color=cmap.colors,
                # child_group=Group(
                    # key_path=['seed'],
                    # readable_name='seed',
                    # operations=[Operation(np.min, 'min'), Operation(np.mean, 'mean'), Operation(np.max, 'max')],
                    # child_group=MetricsDef(
                        # step_key='time/total_timesteps',
                        # min_steps=int(1e6),
                        # keys_to_plot=['rollout/ep_rew_mean'],
                        # dtype=np.float16,
                    # )
                # )
            # )
        # )

    assert os.path.isdir(args.dir), f"Path {args.dir} is not a directory!"
    for exp_parent_dir in os.scandir(args.dir):
        if os.path.isdir(exp_parent_dir):
            for exp_dir in os.scandir(exp_parent_dir):
                log_prefix = f"{exp_dir.path}: \n\t"
                wandb_run_files = [fn for fn in os.listdir(os.path.join(exp_dir, 'wandb')) if fn.startswith("run")]
                assert len(wandb_run_files) == 1
                with open(os.path.join(exp_dir, 'wandb', wandb_run_files[0], 'files', 'config.yaml'), 'r') as file:
                    config = yaml.safe_load(file)
                path_to_progress_csv = os.path.join(exp_dir, 'progress.csv')
                try:
                    data = pd.read_csv(path_to_progress_csv)
                    config['data'] = data
                    err = group.add_run(config)
                    if err is not None:
                        print(log_prefix + err)
                except pandas.errors.EmptyDataError:
                    print(log_prefix + "progress.csv was empty. Skipping.")
                except KeyError as e:
                    print(log_prefix + f"Could not find keys {e.args[0]} in config.yaml. Skipping.")
    del data, config, exp_dir, file, wandb_run_files

    print(group)
    group.apply_operations()
    with open(os.path.join(args.dir, f"{group.key}.pickle"), 'wb') as f:
        pickle.dump(group, f, pickle.HIGHEST_PROTOCOL)
