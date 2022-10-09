import os
import csv
from typing import List, Dict, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import yaml
import copy
from dataclasses import dataclass


@dataclass
class Operation:
    op: Callable
    name: str

    def __repr__(self):
        return self.name


@dataclass
class PlotDef:
    key_regexp: Tuple
    color_alpha: float = 1.0
    legend_name: Optional[str] = None


@dataclass
class MetricsDef:
    keys_to_plot: list
    step_key: str = 'steps'


class Metrics:
    def __init__(self, config: MetricsDef):
        assert config.step_key not in config.keys_to_plot
        self.__config = config
        self.steps = None
        self.__metrics = {}

    @property
    def config(self):
        return self.__config

    def add(self, df):
        if isinstance(df, pd.DataFrame):
            df = dict(df)
            df = {k: np.asarray(v) for k, v in df.items()}
        df_steps = df[self.__config.step_key]
        min_of_max_steps = df_steps.max()
        if self.steps is None:
            self.steps = np.arange(df_steps.max())
        else:
            min_of_max_steps = np.min((self.steps.max(), df_steps.max()))
            self.steps = np.arange(min_of_max_steps)
        for key in self.__config.keys_to_plot:
            interp_key = np.interp(np.arange(df_steps.max()), df_steps, df[key])
            if key not in self.__metrics.keys():
                self.__metrics[key] = interp_key
            else:
                self.__metrics[key] = self.__metrics[key][..., :min_of_max_steps]
                interp_key = interp_key[..., :min_of_max_steps]
                self.__metrics[key] = np.vstack((self.__metrics[key], interp_key))
            assert self.__metrics[key].shape[-1] == self.steps.shape[-1]

    def keys(self):
        return self.__config.keys_to_plot

    def aggregate_metrics(self):
        return self

    @property
    def metrics(self):
        return self.__metrics

    def __getitem__(self, item):
        return self.__metrics[item]

    def __repr__(self):
        return f"{'empty' if self.steps is None else 'filled'} {', '.join([k for k in self.__metrics.keys()])}"


class Group:
    def __init__(self, key_path, color=None, filter: Optional[List[str]] = None, child_group: Optional = None,
                 operations: Optional[List[Operation]] = None):
        if isinstance(key_path, list):
            self.__key_path = []
            for key in key_path:
                if key == 'value':
                    continue
                self.__key_path.append(key)
        else:
            self.__key_path = [key_path]
        self.__filter = filter
        self.__color = color
        self.__child_group = child_group
        self.__subgroups = {}
        self.__operations = operations
        self.__operation_results = None

    @staticmethod
    def get_value(d):
        if isinstance(d, dict) and 'value' in d.keys():
            val = d['value']
        else:
            val = d
        return val

    @property
    def operations(self):
        return self.__operations

    @property
    def operation_results(self):
        return self.__operation_results

    @property
    def color(self):
        return self.__color

    @property
    def subgroups(self):
        return self.__subgroups

    @property
    def key(self):
        return self.__key_path[-1]

    @property
    def data_count(self):
        if isinstance(self.__child_group, Group):
            return np.sum([g.data_count for g in self.__subgroups.values()])
        else:
            return len(self.__subgroups)

    def add_run(self, run: Dict):
        assert 'data' in run.keys()
        key = self.follow_key_path(run)
        if key not in self.__subgroups.keys():
            if isinstance(self.__child_group, Group):
                self.__subgroups[key]: Group = copy.deepcopy(self.__child_group)
            else:
                self.__subgroups[key] = Metrics(self.__child_group)
            self.__subgroups = {k: v for k, v in sorted(self.__subgroups.items(), key=lambda x: x[0])}
        if isinstance(self.__child_group, Group):
            self.__subgroups[key].add_run(run)
        else:
            # A Metrics object should be appended! copy.deepcopy(self.__child_group)
            self.__subgroups[key].add(run['data'])

    def follow_key_path(self, d: Dict):
        val = d
        for k in self.__key_path:
            val = self.get_value(val[k])
        if self.__filter is not None:
            assert isinstance(val, str)
            new_val = None
            for f in self.__filter:
                index = val.find(f)
                if index != -1:
                    new_val = f
                    break
            assert new_val is not None, f"None of the filters {self.__filter} was found in {val}"
            val = new_val
        if val is None:
            val = '<none>'
        return val

    def apply_operations(self):
        assert self.__operation_results is None
        if self.__operations is None:
            assert self.__subgroups is not None, "No operations are defined anywhere"
            for g in self.__subgroups.values():
                g.apply_operations()
        else:
            m = self.aggregate_metrics()
            new_metrics = {}
            for k in m.keys():
                for op in self.__operations:
                    new_metrics[f"{op.name}_{k}"] = op.op(m[k], axis=0)
            m.metrics.update(new_metrics)
            self.__operation_results = m

    def plot(self, plot_options: List[PlotDef], group_mapping: Dict = None):
        if any(have_res := [g.operation_results is not None for g in self.__subgroups.values()]):
            assert all(have_res)
            fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
            for g_index, (g_name, g) in enumerate(self.__subgroups.items()):
                res = g.operation_results.metrics
                x = g.operation_results.steps
                for opt in plot_options:
                    plot_kwargs = {
                        'color': self.color[g_index],
                        'alpha': opt.color_alpha,
                        'label': group_mapping[g_name] if g_name in group_mapping.keys() else g_name,
                    }
                    if len(opt.key_regexp) == 1:
                        y_pattern = opt.key_regexp[0]
                        y_key = None
                        for k in res.keys():
                            if y_key is None and y_pattern.match(k) is not None:
                                y_key = k
                        ax1.plot(x, res[y_key], **plot_kwargs)
                    elif len(opt.key_regexp) == 2:
                        y1_pattern = opt.key_regexp[0]
                        y1_key = None
                        y2_pattern = opt.key_regexp[1]
                        y2_key = None
                        for k in res.keys():
                            if y1_key is None and y1_pattern.match(k) is not None:
                                y1_key = k
                            if y2_key is None and y2_pattern.match(k):
                                y2_key = k
                        assert y1_key is not None, f"Pattern {y1_pattern.pattern} didn't match any key " \
                                                   f"in {', '.join([k for k in res.keys()])}"
                        assert y2_key is not None, f"Pattern {y2_pattern.pattern} didn't match any key " \
                                                   f"in {', '.join([k for k in res.keys()])}"
                        ax1.fill_between(x=x, y1=res[y1_key], y2=res[y2_key], **plot_kwargs)
                    else:
                        raise NotImplementedError(f"Don't know what to do with {len(opt.key_regexp)} keys to plot")
                print(1)
            ax1.legend()
            print(2)




        else:
            [g.plot(plot_options) for g in self.__subgroups.values()]

    def aggregate_metrics(self) -> Metrics:
        metrics = [g.aggregate_metrics() for g in self.__subgroups.values()]
        new_keys = list(set().union(*[set(m.metrics.keys()) for m in metrics]))
        agg_metric = Metrics(MetricsDef(step_key='steps', keys_to_plot=new_keys))
        for m in metrics:
            agg_metric.add({'steps': m.steps, **m.metrics})
        return agg_metric

    def metrics(self):
        raise NotImplemented()

    def __repr__(self):
        if isinstance(self.__child_group, Group):
            return f"{self.key}: {self.__subgroups}"
        else:
            return f"{self.key}: {self.__subgroups.keys()}"


if __name__ == "__main__":
    # mpl.use('Agg')
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
    jf_group.apply_operations()
    jf_group.plot(
        plot_options=[
            PlotDef(key_regexp=(re.compile('^min_roll'), re.compile('^max_roll')), color_alpha=0.3),
            PlotDef(key_regexp=(re.compile('^mean_roll'),), legend_name='Episode reward'),
        ],
        group_mapping={
            '<none>': 'MLP policy',
        }
    )

    for jf_val, ent_obj_group in jf_group.subgroups.items():
        for ent_obj_val, agg_group in ent_obj_group.subgroups.items():
            print(1)

    dir_re = ['recursive_aux_no_grad', 'recursive',
              'huber', 'huber_hack',
              'naive']

    child_dirs = os.listdir(args.dir)
    recursive_aux_no_grad_set = set(filter(re.compile('^recursive_aux_no_grad').match, child_dirs))
    recursive_set = set(filter(re.compile('^recursive').match, child_dirs))
    huber_set = set(filter(re.compile('^huber').match, child_dirs))
    huber_hack_set = set(filter(re.compile('^huber_hack').match, child_dirs))
    naive_set = set(filter(re.compile('^naive').match, child_dirs))

    recursive_set -= recursive_aux_no_grad_set
    huber_set -= huber_hack_set

    def color_and_dirs(color, dirs):
        return {'color': color, 'dirs': dirs}

    cmap = mpl.cm.get_cmap('tab10')
    objs_to_plot = {
        'recursive': color_and_dirs(cmap.colors[0], recursive_set),
        'huber': color_and_dirs(cmap.colors[1], huber_set),
        'huber_hack': color_and_dirs(cmap.colors[2], huber_hack_set),
        'naive': color_and_dirs(cmap.colors[3], naive_set),
    }
    assert all(len(i['dirs']) > 0 for i in objs_to_plot.values())
    nr_seeds = [len(i['dirs']) for i in objs_to_plot.values()]
    assert all(i == nr_seeds[0] for i in nr_seeds)

    metrics_operations = {
        'min': np.min,
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
    }

    def re_and_empty_list(regexp: list, op: str, ylim=None, plot_kwargs=None):
        if plot_kwargs is None:
            plot_kwargs = {}
        assert op != ''
        return {
            'regexp': [re.compile(r) for r in regexp],
            'op': op,  # String must contain one of mean, min, max, median
            'ylim': ylim,
            'plot_kwargs': plot_kwargs,
            # A key for every objective to plot will be added
        }

    plots = {
        # plot title: regular expression to match saved metric keys on
        'Root sum node naive entropy approximation':
            re_and_empty_list(['naive_root_entropy'], 'mean', ylim=(None, 120)),
        'Root sum node recursive entropy approximation': re_and_empty_list(['recursive_ent_approx'], 'mean'),
        'Root sum node Huber entropy lower bound': re_and_empty_list(['huber_entropy_LB'], 'mean'),
        'Root sum node weight entropy': re_and_empty_list(['4/rep0/weight_entropy'], 'mean'),
        'Leaf entropies': re_and_empty_list(
            ['0/rep./node_entropy/min', '0/rep./node_entropy/mean', '0/rep./node_entropy/max'],
            'min/median/max',
        ),
        'Layer 2 sum node weight entropies': re_and_empty_list(
            ['2/rep./weight_entropy/min', '2/rep./weight_entropy/mean', '2/rep./weight_entropy/max'],
            'min/median/max',
        ),
    }
    for obj_name in objs_to_plot.keys():
        for plot_title in plots.keys():
            plots[plot_title][obj_name] = {'raw': []}

    for obj_name, obj_dict in objs_to_plot.items():
        nr_seeds = len(obj_dict['dirs'])
        config = None
        metric_keys = None
        for dir_name in obj_dict['dirs']:
            cwd = os.path.join(args.dir, dir_name)
            assert os.path.exists(cwd)
            print(f"Reading from {cwd}")

            metrics = None
            for filename in os.listdir(cwd):
                filename = os.fsdecode(filename)
                if filename.endswith('.csv') and filename.startswith('config'):
                    with open(os.path.join(cwd, filename)) as f:
                        reader = csv.DictReader(f)
                        curr_config = [r for r in reader][0]
                if filename.endswith('.npz') and filename.startswith('metrics'):
                    metrics = np.load(os.path.join(cwd, filename), allow_pickle=True)
                    keys = [i for i in metrics]
                    assert len(keys) == 1, f"There should only be one key, but there were {len(keys)}"
                    metrics = metrics[keys[0]].tolist()
                    if metric_keys is None:
                        metric_keys = list(metrics.keys())
                    else:
                        assert set(metric_keys) == set(metrics.keys())
            assert metrics is not None
            if config is None:
                config = curr_config
                assert config['RATSPN_F'] == '4', "If plots should be generated for an SPN with " \
                                                  "other than 4 leaf features, you need to adapt " \
                                                  "the regular expressions. "
            for plot_contents in plots.values():
                matching_metrics = [set(filter(r.match, metric_keys)) for r in plot_contents['regexp']]
                assert len(matching_metrics[0].union(*matching_metrics[1:])) == np.sum([len(a) for a in matching_metrics])
                matching_metrics = list(matching_metrics[0].union(*matching_metrics[1:]))
                plot_contents[obj_name]['raw'] += [metrics[key] for key in matching_metrics]

        for plot_title, plot_contents in plots.items():
            plot_contents[obj_name]['raw'] = np.vstack(plot_contents[obj_name]['raw'])
            for op in metrics_operations.keys():
                if op in plot_contents['op']:
                    plot_contents[obj_name][op] = metrics_operations[op](plot_contents[obj_name]['raw'], axis=0)

    for plot_title, data_dict in plots.items():
        fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
        if data_dict['ylim'] is not None:
            ax1.set_ylim(*data_dict['ylim'])
        for obj_name, obj_dict in objs_to_plot.items():
            obj_data = data_dict[obj_name]
            col = obj_dict['color']
            if 'min' in obj_data.keys() and 'max' in obj_data.keys():
                ax1.fill_between(range(len(obj_data['min'])), obj_data['min'], obj_data['max'],
                                 color=col, alpha=0.5)
            avg = obj_data['mean'] if 'mean' in obj_data.keys() else obj_data['median']
            ax1.plot(avg, color=col, label=obj_name)

            ax1.legend()
        fig.suptitle(plot_title)
        fig.show()
        print(1)
