import os
import matplotlib as mpl
import pickle

from classes import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, required=True,
                        help='Directory containing experiment directories (the project directory)')
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
    cmap = mpl.cm.get_cmap('tab10')
    jf_group = Group(
            key_path=['joint_failure_prob'],
            child_group=Group(
                key_path=['policy_kwargs', 'actor_cspn_args', 'entropy_objective'],
                color=cmap.colors,
                child_group=Group(
                    key_path=['seed'],
                    operations=[Operation(np.min, 'min'), Operation(np.mean, 'mean'), Operation(np.max, 'max')],
                    child_group=MetricsDef(step_key='time/total_timesteps', keys_to_plot=['rollout/ep_rew_mean'])
                )
            )
        )

    assert os.path.isdir(args.dir), f"Path {args.dir} is not a directory!"
    for exp_parent_dir in os.scandir(args.dir):
        if os.path.isdir(exp_parent_dir):
            for exp_dir in os.scandir(exp_parent_dir):
                wandb_run_files = [fn for fn in os.listdir(os.path.join(exp_dir, 'wandb')) if fn.startswith("run")]
                assert len(wandb_run_files) == 1
                with open(os.path.join(exp_dir, 'wandb', wandb_run_files[0], 'files', 'config.yaml'), 'r') as file:
                    config = yaml.safe_load(file)
                data = pd.read_csv(os.path.join(exp_dir, 'progress.csv'))
                config['data'] = data
                jf_group.add_run(config)
    del data, config, exp_dir, file, wandb_run_files

    print(jf_group.data_count)
    with open(os.path.join(args.dir, f"{jf_group.name}.pickle"), 'wb') as f:
        pickle.dump(jf_group, f, pickle.HIGHEST_PROTOCOL)
