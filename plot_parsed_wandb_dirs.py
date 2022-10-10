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

    fig_options = {
        'save_path': save_path,
        'x_axis_label': 'Steps',
        'y_axis_label': 'Reward',
        'x_axis_range': [None, None],
        'legend_loc': 'lower right',
        'size': (7, 7),
        'dpi': 200,
        # 'upper left', 'upper right', 'lower left', 'lower right',
        # 'upper center', 'lower center', 'center left', 'center right'
    }
    plot_defs = [
        PlotDef(
            line_defs=[
                FillDef(key_regexp=(re.compile('^min_roll'), re.compile('^max_roll')),
                        color_alpha=0.3, add_to_legend=False),
                LineDef(key_regexp=re.compile('^mean_roll'), add_to_legend=True),
            ],
            title='Episode Reward',
            legend_title='Policy',
            figure_options=fig_options,
        ),
        PlotDef(
            line_defs=[
                FillDef(key_regexp=(re.compile('^min_train/huber'), re.compile('^max_train/huber')),
                        color_alpha=0.3, add_to_legend=False),
                LineDef(key_regexp=re.compile('^mean_train/huber'), add_to_legend=True),
            ],
            title='Huber Entropy LB',
            legend_title='Policy',
            figure_options=fig_options,
        ),
        PlotDef(
            line_defs=[
                FillDef(key_regexp=(re.compile('^min_train/recur'), re.compile('^max_train/recur')),
                        color_alpha=0.3, add_to_legend=False),
                LineDef(key_regexp=re.compile('^mean_train/recur'), add_to_legend=True),
            ],
            title='Recursive Entropy Approx',
            legend_title='Policy',
            figure_options=fig_options,
        ),
        PlotDef(
            line_defs=[
                FillDef(key_regexp=(re.compile('^min_train/naive'), re.compile('^max_train/naive')),
                        color_alpha=0.3, add_to_legend=False),
                LineDef(key_regexp=re.compile('^mean_train/naive'), add_to_legend=True),
            ],
            title='Naive Entropy Approx',
            legend_title='Policy',
            figure_options=fig_options,
        ),
        PlotDef(
            line_defs=[
                FillDef(key_regexp=(re.compile('^min_train/actor'), re.compile('^max_train/actor')),
                        color_alpha=0.3, add_to_legend=False),
                LineDef(key_regexp=re.compile('^mean_train/actor'), add_to_legend=True),
            ],
            title='Actor Entropy',
            legend_title='Policy',
            figure_options=fig_options,
        ),
    ]
    [group.plot(
        plot_option=p,
        title_prefix=p.title,
        group_mapping={
            '<none>': 'MLP Policy',
            'augmented_huber': 'SPN (Augmented Huber)',
            'huber': 'SPN (Huber)',
            'naive': 'SPN (Naive)',
            'recursive': 'SPN (Recursive)',
        }
    ) for p in plot_defs]
