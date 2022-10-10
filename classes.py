import os
from typing import List, Dict, Optional, Tuple, Any, Callable, Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import matplotlib as mpl
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
    min_steps: int = None
    step_key: str = 'steps'
    dtype: Type = np.float32


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
        existing_max_step = self.steps[-1] if self.steps is not None else None
        df_steps = df[self.__config.step_key]
        df_step_max = df_steps[-1]
        pad_len = None
        if self.steps is None:
            self.steps = np.arange(df_step_max+1)
        else:
            pad_len = np.abs(existing_max_step - df_step_max)
            new_max_step = np.max((existing_max_step, df_step_max))
            self.steps = np.arange(new_max_step+1)
        for key in self.__config.keys_to_plot:
            interp_key = np.interp(np.arange(df_step_max+1), df_steps, df[key])
            interp_key = np.array(interp_key, dtype=self.__config.dtype)
            if key not in self.__metrics.keys():
                self.__metrics[key] = interp_key
            else:
                m_key = self.__metrics[key]
                if df_step_max < existing_max_step:
                    pad = m_key[..., -pad_len:]
                    if len(m_key.shape) > 1:
                        pad = pad.mean(axis=0)
                    interp_key = np.hstack((interp_key, pad))
                elif df_step_max > existing_max_step:
                    pad = interp_key[-pad_len:]
                    if len(m_key.shape) > 1:
                        pad = pad[np.newaxis].repeat(m_key.shape[0], axis=0)
                    m_key = np.hstack((m_key, pad))
                self.__metrics[key] = np.vstack((m_key, interp_key))
            assert self.__metrics[key].shape[-1] == self.steps.shape[-1]
        return None

    def keys(self):
        return self.__config.keys_to_plot

    def aggregate_metrics(self):
        return self

    @property
    def metrics(self):
        return self.__metrics

    @property
    def empty(self):
        return self.steps is None

    def __getitem__(self, item):
        return self.__metrics[item]

    def __repr__(self):
        return f"{'empty' if self.empty else 'filled'} {', '.join([k for k in self.__metrics.keys()])}"


def build_group_from_kwargs(d: Dict):
    if 'child_group' in d.keys():
        new_ops = None
        if (ops := d.get('operations')) is not None:
            new_ops = []
            for op in ops:
                if op == 'min':
                    new_ops.append(Operation(np.min, 'min'))
                elif op == 'mean':
                    new_ops.append(Operation(np.mean, 'mean'))
                elif op == 'max':
                    new_ops.append(Operation(np.max, 'max'))
                else:
                    raise NotImplementedError()
        return Group(
            key_path=d.get('key_path'),
            readable_name=d.get('readable_name'),
            color=mpl.cm.get_cmap(col).colors if (col := d.get('color')) is not None else None,
            operations=new_ops,
            child_group=build_group_from_kwargs(d.get('child_group'))
        )
    else:
        data_type = d.get('dtype')
        if data_type is not None:
            if 'float64' in data_type:
                data_type = np.float64
            if 'float32' in data_type:
                data_type = np.float32
            if 'float16' in data_type:
                data_type = np.float16
            else:
                raise NotImplementedError()
        return MetricsDef(
            step_key=d.get('step_key'),
            min_steps=int(float(s)) if (s := d.get('min_steps')) is not None else None,
            keys_to_plot=d.get('keys_to_plot'),
            dtype=data_type,
        )


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
            return self.__subgroups[key].add_run(run)
        else:
            return self.__subgroups[key].add(run['data'])

    def follow_key_path(self, d: Dict):
        val = d
        try:
            for k in self.__key_path:
                val = self.get_value(val[k])
        except KeyError:
            raise KeyError(str(self.__key_path))
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
                min_metric = None
                max_metric = None
                for op in self.__operations:
                    res = op.op(m[k], axis=0)
                    if op.name == 'min':
                        min_metric = res
                    if op.name == 'max':
                        max_metric = res
                    new_metrics[f"{op.name}_{k}"] = res
                if len(m[k].shape) > 1 and m[k].shape[0] > 1 and min_metric is not None and max_metric is not None:
                    both_same = min_metric == max_metric
                    for new_k, v in new_metrics.items():
                        if k in new_k:
                            new_metrics[new_k] = v[~both_same]
                m.metrics.pop(k)
            m.metrics.update(new_metrics)
            self.__operation_results = m

    def plot(self, plot_options: List[PlotDef], title_prefix: str = None,
             figure_options: Dict = None, group_mapping: Dict = None):
        if any(have_res := [g.operation_results is not None for g in self.__subgroups.values()]):
            assert all(have_res)
            fig, (ax1) = plt.subplots(
                1,
                figsize=size if (size := figure_options.get('size')) else (10, 10),
                dpi=dpi if (dpi := figure_options.get('dpi')) else 200,
            )
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
                        ax1.plot(np.arange(res[y_key].shape[0]), res[y_key], **plot_kwargs)
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
                        assert res[y1_key].shape == res[y2_key].shape
                        ax1.fill_between(x=np.arange(res[y1_key].shape[0]),
                                         y1=res[y1_key], y2=res[y2_key], **plot_kwargs)
                    else:
                        raise NotImplementedError(f"Don't know what to do with {len(opt.key_regexp)} keys to plot")
            ax1.legend(title='Policy', loc='best' if (loc := figure_options.get('legend_loc')) is None else loc)
            ax1.set_title(title_prefix)
            if (x_axis_label := figure_options.get('x_axis_label')) is not None:
                ax1.set_xlabel(x_axis_label)
            if (y_axis_label := figure_options.get('y_axis_label')) is not None:
                ax1.set_ylabel(y_axis_label)
            if (x_axis_range := figure_options.get('x_axis_range')) is not None:
                assert len(x_axis_range) == 2
                ax1.set_xbound(x_axis_range[0], x_axis_range[1])
            fig.tight_layout()
            if (path := figure_options.get('save_path')) is not None:
                file_name = title_prefix.lower().replace('.', '_').replace(' ', '').replace(':', '') + '.png'
                fig.savefig(os.path.join(path, file_name))
            else:
                fig.show()
            fig.clf()
        else:
            return [g.plot(
                plot_options=plot_options,
                title_prefix=f"{title_prefix} - {self.name}: {k}",
                figure_options=figure_options,
                group_mapping=group_mapping
            ) for k, g in self.__subgroups.items()]

    def aggregate_metrics(self) -> Metrics:
        metrics = [g.aggregate_metrics() for g in self.__subgroups.values()]
        new_keys = list(set().union(*[set(m.metrics.keys()) for m in metrics]))
        agg_metric = Metrics(MetricsDef(step_key='steps', keys_to_plot=new_keys))
        [agg_metric.add({'steps': m.steps, **m.metrics}) for m in metrics if not m.empty]
        return agg_metric

    def metrics(self):
        raise NotImplemented()

    def __repr__(self):
        if isinstance(self.__child_group, Group):
            child_repr = '\n' + '\n'.join([f'{k}:' + f'\n{g.__repr__()}'.replace('\n', '\n\t')
                                    for k, g in self.__subgroups.items()])
            return f"{self.key}:" + child_repr.replace('\n', '\n\t')
        else:
            return f"{self.key}: {', '.join(str(k) for k in self.__subgroups.keys())}"
