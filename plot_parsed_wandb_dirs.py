import os.path
import re
import pickle

from classes import *

if __name__ == "__main__":
    # mpl.use('Agg')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True, help='Path to pickled file from parse_wandb_dirs.py')
    args = parser.parse_args()

    assert os.path.exists(args.path)
    with open(args.path, 'rb') as f:
        group = pickle.load(f)

    save_path = os.path.join(os.path.split(args.path)[0], 'plots')
    os.makedirs(save_path, exist_ok=True)

    plots = group.plot(
        plot_options=[
            PlotDef(key_regexp=(re.compile('^min_roll'), re.compile('^max_roll')), color_alpha=0.3, add_to_legend=False),
            PlotDef(key_regexp=(re.compile('^mean_roll'),), add_to_legend=True),
        ],
        title_prefix='Episode Reward',
        figure_options={
            'save_path': save_path,
            'x_axis_label': 'Steps',
            'y_axis_label': 'Reward',
            'x_axis_range': [None, int(2.4e6)],
            'legend_loc': 'lower right',
            'size': (7, 7),
            'dpi': 200,
            # 'upper left', 'upper right', 'lower left', 'lower right',
            # 'upper center', 'lower center', 'center left', 'center right'
        },
        group_mapping={
            '<none>': 'MLP Policy',
            'augmented_huber': 'SPN (Augmented Huber)',
            'huber': 'SPN (Huber)',
            'naive': 'SPN (Naive)',
            'recursive': 'SPN (Recursive)',
        }
    )
