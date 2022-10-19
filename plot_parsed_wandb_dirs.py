import os.path
import re
import pickle

from classes import *

if __name__ == "__main__":
    # mpl.use('Agg')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)
        assert all([k in config.keys() for k in ['plot', 'experiment_dir', 'pickle_save_path', 'plot_save_path']])
        exp_dir = config.get('experiment_dir')
        assert os.path.isdir(exp_dir), f"Path {exp_dir} is not a directory!"
        os.chdir(exp_dir)
        pickle_path = config.get('pickle_save_path')
        save_path = config.get('plot_save_path')
        os.makedirs(save_path, exist_ok=True)
    assert os.path.exists(pickle_path)
    with open(pickle_path, 'rb') as f:
        group = pickle.load(f)

    plot_config = config.get('plot')
    plot_defs = build_plot_defs_from_config(plot_config)
    [group.plot(
        plot_option=p,
        title_prefix=p.title,
        fig_save_path=save_path,
        legend_options=plot_config.get('legend'),
    ) for p in plot_defs]
