import os
from typing import List, Dict, Optional, Tuple, Callable, Type, Pattern

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
from slugify import slugify


def get_key_path_value(key_path, d):
    if not key_path:
        return d
    key = key_path.pop(0)
    return get_key_path_value(key_path, d[key])


@dataclass
class Operation:
    op: Callable
    name: str

    def __repr__(self):
        return self.name


@dataclass
class BaseLineDef:
    color_alpha: float = 1.0
    add_to_legend: bool = True


@dataclass
class LineDef(BaseLineDef):
    key_regexp: Pattern[str] = None


@dataclass
class FillDef(BaseLineDef):
    key_regexp: Tuple[Pattern, Pattern] = None


@dataclass
class PlotDef:
    line_defs: List[BaseLineDef] = None
    title: str = None
    figure_options: Dict = None


@dataclass
class MetricsDef:
    keys_to_plot: List[Pattern]
    min_steps: int = None
    step_key: str = 'steps'
    dtype: Type = np.float32
    step_delta: int = 1000


class Metrics:
    def __init__(self, config: MetricsDef):
        assert config.step_key not in config.keys_to_plot
        self.__config = config
        self.__metrics = {}

    @property
    def config(self):
        return self.__config

    @property
    def step_delta(self):
        return self.config.step_delta

    @property
    def max_step(self):
        if self.metrics == {}:
            return None
        nr_data_points = list(self.metrics.values())[0].shape[-1]
        return (nr_data_points - 1) * self.step_delta

    @property
    def steps(self):
        max_step = self.max_step
        if max_step is None:
            return None
        steps = np.arange(0, max_step + self.step_delta, self.step_delta)
        assert max_step == steps[-1]
        return steps

    def add(self, df):
        if isinstance(df, pd.DataFrame):
            df = dict(df)
            df = {k: np.asarray(v)[np.newaxis, :] for k, v in df.items()}
            df_steps = df[self.__config.step_key]
            if len(df_steps.shape) == 2:
                df_steps = df_steps[0]
            df_step_max = df_steps[-1]
        elif isinstance(df, Metrics):
            df_steps = df.steps
            df_step_max = df.max_step
        else:
            raise NotImplementedError("df must either be a pandas DataFrame or a Metrics object.")

        self_max_step = self.max_step
        if self_max_step is None:
            self_max_step = df_step_max

        assert self.step_delta is not None, "step_delta was not set in config!"
        pad_len = int((self_max_step - df_step_max) / self.step_delta)
        assert pad_len == 0 if isinstance(df, pd.DataFrame) else True
        pad_df = True if pad_len > 0 else False
        pad_metric = True if pad_len < 0 else False
        pad_len = abs(pad_len)

        new_max_step = max(self_max_step, df_step_max)
        new_steps = np.arange(0, new_max_step + self.step_delta, self.step_delta)

        for key_regex in self.__config.keys_to_plot:
            keys = [m.string for k in df.keys() if (m := key_regex.match(k)) is not None]
            for key in keys:
                assert len(df[key].shape) == 2
                df_interp = np.vstack([np.interp(new_steps, df_steps, df[key][i]) for i in range(df[key].shape[0])])
                df_interp = df_interp.astype(self.config.dtype)
                if key_regex.pattern not in self.metrics.keys():
                    self.metrics[key_regex.pattern] = df_interp
                else:
                    m_key = self.metrics[key_regex.pattern]
                    if pad_df:
                        pad = m_key[..., -pad_len:]
                        pad = pad.mean(axis=0)
                        df_interp[..., -pad_len:] = pad
                    elif pad_metric:
                        pad = df_interp[..., -pad_len:]
                        pad = pad.repeat(m_key.shape[0] // df_interp.shape[0], axis=0)
                        m_key = np.hstack((m_key, pad))
                    self.__metrics[key_regex.pattern] = np.vstack((m_key, df_interp))
                assert self.metrics[key_regex.pattern].shape[-1] == self.steps.shape[-1]
                # existing_max_step = self.steps[-1]
                # df_step_max = existing_max_step
        # assert all metrics have same len
        return None

    def aggregate_metrics(self):
        return self

    @property
    def metrics(self):
        return self.__metrics

    def keys(self):
        return self.metrics.keys()

    @property
    def empty(self):
        return self.steps is None

    def __getitem__(self, item):
        return self.metrics[item]

    def __repr__(self):
        return f"{'empty' if self.empty else 'filled'} {', '.join([k for k in self.metrics.keys()])}"


class Group:
    def __init__(self, key_path, readable_name, child_group: Optional = None,
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
    def subgroups(self):
        return self.__subgroups

    @property
    def key_path(self):
        return self.__key_path

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
        key = get_key_path_value(self.key_path.copy(), run)
        if key is None:
            key = 'None'
        if isinstance(key, list):
            key = tuple(key)
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

    def apply_operations(self):
        assert self.__operation_results is None
        if self.__operations is None:
            assert self.__subgroups is not None, "No operations are defined anywhere"
            for g in self.__subgroups.values():
                g.apply_operations()
        else:
            m = self.aggregate_metrics()
            new_metrics = {}
            keys = list(m.keys())
            for k in keys:
                min_metric = None
                max_metric = None
                for op in self.__operations:
                    res = op.op(m[k], axis=0)
                    if op.name == 'min':
                        min_metric = res
                    if op.name == 'max':
                        max_metric = res
                    new_metrics[f"{op.name}_{k}"] = res
                m.metrics.pop(k)
            m.metrics.update(new_metrics)
            self.__operation_results = m

    def plot(self, plot_option: PlotDef, title_prefix: str = None, fig_save_dir: str = None,
             legend_options: Dict = None):
        if any(have_res := [g.operation_results is not None for g in self.__subgroups.values()]):
            assert all(have_res)
            opt = plot_option
            fig_opt = opt.figure_options
            fig, (ax1) = plt.subplots(
                1,
                figsize=size if (size := fig_opt.get('size')) else (10, 10),
                dpi=dpi if (dpi := fig_opt.get('dpi')) else 200,
            )
            for g_index, (g_name, g) in enumerate(self.__subgroups.items()):
                if (ign := fig_opt.get('ignore_keys')) is not None and g_name in ign:
                    continue
                res = g.operation_results
                for line_def in opt.line_defs:
                    g_label = None
                    g_color = None
                    if (leg_map := legend_options.get('mapping')) is not None:
                        if g_name in leg_map.keys():
                            g_color = leg_map[g_name][1]
                            col_map, col_map_ind = g_color.split(':')
                            g_color = mpl.cm.get_cmap(col_map).colors[int(col_map_ind)]
                        if line_def.add_to_legend:
                            g_label = leg_map[g_name][0] if g_name in leg_map.keys() else g_name
                    plot_kwargs = {
                        'color': g_color,
                        'alpha': line_def.color_alpha,
                        'label': g_label,
                    }
                    if isinstance(line_def, LineDef):
                        y_pattern = line_def.key_regexp
                        y_key = [m.string for k in res.keys() if (m := y_pattern.match(k)) is not None]
                        if len(y_key) == 0:
                            continue
                        assert len(y_key) == 1, f"Found {len(y_key)} key matches for regex {y_pattern.pattern}"
                        y_key = y_key[0]
                        x = np.arange(res[y_key].shape[0]) * res.step_delta
                        ax1.plot(x, res[y_key], **plot_kwargs)
                    elif isinstance(line_def, FillDef):
                        y1_pattern = line_def.key_regexp[0]
                        y1_key = [m.string for k in res.keys() if (m := y1_pattern.match(k)) is not None]
                        if len(y1_key) == 0:
                            continue
                        assert len(y1_key) == 1, f"Found {len(y1_key)} key matches for regex {y1_pattern.pattern}"
                        y1_key = y1_key[0]
                        y2_pattern = line_def.key_regexp[1]
                        y2_key = [m.string for k in res.keys() if (m := y2_pattern.match(k)) is not None]
                        if len(y2_key) == 0:
                            continue
                        assert len(y2_key) == 1, f"Found {len(y2_key)} key matches for regex {y2_pattern.pattern}"
                        y2_key = y2_key[0]
                        assert res[y1_key].shape == res[y2_key].shape
                        x = np.arange(res[y1_key].shape[0]) * res.step_delta
                        ax1.fill_between(x=x, y1=res[y1_key], y2=res[y2_key], **plot_kwargs)
                    else:
                        raise NotImplementedError()
            if len(ax1.lines) == 0:
                print(f"No parsed data found to plot \n\t{title_prefix}\n"
                      f"Did you forget to add the metrics to keys_to_plot?")
                return
            ax1.legend(title=legend_options.get('title'), loc=legend_options.get('loc'))
            ax1.set_title(title_prefix)
            if (x_axis_label := fig_opt.get('x_axis_label')) is not None:
                ax1.set_xlabel(x_axis_label)
            if (y_axis_label := fig_opt.get('y_axis_label')) is not None:
                ax1.set_ylabel(y_axis_label)
            if (x_axis_range := fig_opt.get('x_axis_range')) is not None:
                assert len(x_axis_range) == 2
                ax1.set_xbound(x_axis_range[0], x_axis_range[1])
            fig.tight_layout()
            if fig_save_dir is not None:
                os.makedirs(fig_save_dir, exist_ok=True)
                file_format = f.replace('.', '') if (f := fig_opt.get('file_format')) is not None else 'png'
                fig_save_path = os.path.join(fig_save_dir, slugify(title_prefix) + '.' + file_format)
                fig.savefig(fig_save_path)
            else:
                fig.show()
            plt.close(fig)
        else:
            return [g.plot(
                plot_option=plot_option,
                title_prefix=f"{title_prefix} - {self.name}: {k}",
                fig_save_dir=os.path.join(fig_save_dir, slugify(title_prefix)),
                legend_options=legend_options,
            ) for k, g in self.__subgroups.items()]

    def aggregate_metrics(self) -> Metrics:
        metrics = [g.aggregate_metrics() for g in self.__subgroups.values()]
        new_keys = list(set().union(*[set(m.config.keys_to_plot) for m in metrics]))
        step_delta = list(set([m.config.step_delta for m in metrics]))
        step_key = list(set([m.config.step_key for m in metrics]))
        assert len(step_delta) == 1
        assert len(step_key) == 1
        step_delta = step_delta[0]
        step_key = step_key[0]
        agg_metric = Metrics(MetricsDef(step_key=step_key, step_delta=step_delta, keys_to_plot=new_keys))
        [agg_metric.add(m) for m in metrics if not m.empty]
        return agg_metric

    def metrics(self):
        raise NotImplemented()

    def __repr__(self):
        child_repr = '\n' + '\n'.join([f'{k}:' + f'\n{g.__repr__()}'.replace('\n', '\n\t')
                                       for k, g in self.__subgroups.items()])
        op_res = f' Operation result - {op.__repr__()}' if (op := self.operation_results) is not None else ''
        return f"{self.key}:{op_res}" + child_repr.replace('\n', '\n\t')
