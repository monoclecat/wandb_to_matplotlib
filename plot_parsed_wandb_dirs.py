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
    with open(args.path) as f:
        group = pickle.load(f)

    group.apply_operations()
    group.plot(
        plot_options=[
            PlotDef(key_regexp=(re.compile('^min_roll'), re.compile('^max_roll')), color_alpha=0.3),
            PlotDef(key_regexp=(re.compile('^mean_roll'),), legend_name='Episode reward'),
        ],
        group_mapping={
            '<none>': 'MLP policy',
        }
    )
