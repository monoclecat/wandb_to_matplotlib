import os
import pickle
from util import build_plot_defs_from_config, parse_config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = parse_config(args.config)
    save_path = config.get('plot_save_path')
    os.makedirs(save_path, exist_ok=True)
    pickle_path = config.get('pickle_save_path')
    assert os.path.exists(pickle_path)
    with open(pickle_path, 'rb') as f:
        group = pickle.load(f)

    plot_config = config.get('plot')
    plot_defs = build_plot_defs_from_config(plot_config)
    [group.plot(
        plot_option=p,
        title_prefix=p.title,
        fig_save_dir=save_path,
        legend_options=plot_config.get('legend'),
    ) for p in plot_defs]
