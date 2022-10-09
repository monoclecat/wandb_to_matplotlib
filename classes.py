from typing import List, Dict, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import re
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
    add_to_legend: bool = True


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
    def __init__(self, key_path, readable_name, color=None, filter: Optional[List[str]] = None, child_group: Optional = None,
                 operations: Optional[List[Operation]] = None):
        if isinstance(key_path, list):
            self.__key_path = []
            for key in key_path:
                if key == 'value':
                    continue
                self.__key_path.append(key)
        else:
            self.__key_path = [key_path]
        self.__name = readable_name
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
    def name(self):
        return self.__name

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

    def plot(self, plot_options: List[PlotDef], figure_options: Optional[Dict], group_mapping: Dict = None):
        if any(have_res := [g.operation_results is not None for g in self.__subgroups.values()]):
            assert all(have_res)
            fig, (ax1) = plt.subplots(1, figsize=(10, 10), dpi=200)
            for g_index, (g_name, g) in enumerate(self.__subgroups.items()):
                res = g.operation_results.metrics
                x = g.operation_results.steps
                for opt in plot_options:
                    if group_mapping is not None and opt.add_to_legend:
                        g_label = group_mapping[g_name] if g_name in group_mapping.keys() else g_name
                    else:
                        g_label = None
                    plot_kwargs = {
                        'color': self.color[g_index],
                        'alpha': opt.color_alpha,
                        'label': g_label,
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
            ax1.legend()
            return fig
        else:
            return [g.plot(plot_options=plot_options, figure_options=figure_options, group_mapping=group_mapping)
                    for g in self.__subgroups.values()]

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
