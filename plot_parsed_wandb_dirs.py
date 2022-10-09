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

    group.apply_operations()
    plots = group.plot(
        plot_options=[
            PlotDef(key_regexp=(re.compile('^min_roll'), re.compile('^max_roll')), color_alpha=0.3, add_to_legend=False),
            PlotDef(key_regexp=(re.compile('^mean_roll'),), add_to_legend=True),
        ],
        figure_options=None,  # TODO Legend loc, plot title, x/y axis labels, x axis range
        group_mapping={
            '<none>': 'MLP policy',
        }
    )
    # TODO plot title is concatenated readable names and values
    [f.show() for f in plots]
    print(1)
