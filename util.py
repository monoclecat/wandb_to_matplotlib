from typing import Dict, List, Callable
import matplotlib as mpl
import numpy as np
import re
import os
import yaml

import pandas as pd

from classes import Group, Operation, MetricsDef, PlotDef, FillDef, LineDef
import wandb
import tqdm


def parse_config(path_to_config: str):
    assert os.path.exists(path_to_config)
    with open(path_to_config, 'r') as file:
        config: dict = yaml.safe_load(file)
        assert all([k in config.keys() for k in ['root_dir', 'pickle_save_path', 'csv_file_name']])
        root_dir = config.get('root_dir')
        assert os.path.isdir(root_dir), f"Path {root_dir} is not a directory!"
        print(f"Changing current working directory to {root_dir}")
        os.chdir(root_dir)
        save_path = config.get('pickle_save_path')
        assert os.path.exists(base_path := os.path.split(save_path)[0]), \
            f"Base path {base_path} of pickle_save_path doesn't exist!"
    return config


def download_run_configs(root_dir: str, wandb_entity: str, wandb_project: str, exclude_run_ids: List = None):
    if exclude_run_ids is None:
        exclude_run_ids = []
    api = wandb.Api()
    runs = api.runs(wandb_entity + '/' + wandb_project)
    for run in runs:
        if run.id in exclude_run_ids:
            continue
        local_run_path = root_dir
        if (group := run.group) is not None:
            local_run_path = os.path.join(local_run_path, group)
        timestamp = run.created_at.replace('-', '').replace(':', '').replace('T', '_')
        local_run_path = os.path.join(local_run_path, run.name, 'wandb', f'run-{timestamp}-{run.id}')

        local_files_path = os.path.join(local_run_path, 'files')
        os.makedirs(local_files_path, exist_ok=True)
        with open(os.path.join(local_files_path, 'config.yaml'), 'w') as f:
            yaml.dump(run.config, f)
    return


def find_wandb_dirs(parent_dir: str) -> List[str]:
    if not os.path.isdir(parent_dir):
        return []
    dir_contents = os.listdir(parent_dir)
    if 'wandb' in dir_contents:
        return [parent_dir]
    else:
        list_of_lists = [find_wandb_dirs(os.path.join(parent_dir, c)) for c in dir_contents]
        flattened_list = [d for d_list in list_of_lists for d in d_list]
        return flattened_list


def without_intermediate_value_keys(d):
    if not isinstance(d, dict):
        return d
    if len(d.keys()) == 2 and 'value' in d.keys():
        return without_intermediate_value_keys(d['value'])
    return {k: without_intermediate_value_keys(v) for k, v in d.items()}


def apply_func_to_metrics(exp_dirs: List[str], config: Dict, func: Callable, **kwargs):
    """

    :param exp_dirs:
    :param config:
    :param func: Must take the kwargs log_prefix, experiment_config, path_to_csv
    :return:
    """
    for exp_dir in exp_dirs:
        log_prefix = f"{exp_dir}: \n\t"
        wandb_run_dir = find_wandb_run_dir_in_exp_dir(exp_dir)
        if wandb_run_dir is None:
            continue
        with open(os.path.join(exp_dir, 'wandb', wandb_run_dir, 'files', 'config.yaml'), 'r') as file:
            experiment_config = yaml.safe_load(file)
            experiment_config = without_intermediate_value_keys(experiment_config)
        path_to_csv = os.path.join(exp_dir, config.get('csv_file_name'))
        func(**locals(), **kwargs)


def find_wandb_run_dir_in_exp_dir(exp_dir: str):
    wandb_run_dir = None
    if os.path.exists(wandb_path := os.path.join(exp_dir, 'wandb')):
        wandb_run_dir = [fn for fn in os.listdir(wandb_path) if fn.startswith("run")]
    if wandb_run_dir is None:
        return wandb_run_dir
    assert len(wandb_run_dir) == 1
    wandb_run_dir = wandb_run_dir[0]
    return wandb_run_dir


def run_id_from_wandb_run_dir(wandb_run_dir: str):
    run_name_components = wandb_run_dir.split('-')
    return run_name_components[-1]


def run_ids_in_exp_dirs(exp_dirs: List):
    run_ids = []
    for exp_dir in exp_dirs:
        wandb_run_dir = find_wandb_run_dir_in_exp_dir(exp_dir)
        if wandb_run_dir is None:
            continue
        run_id = run_id_from_wandb_run_dir(wandb_run_dir)
        run_ids.append(run_id)
    return run_ids


def download_metrics_and_save_as_csv(exp_dirs: List[str], config: Dict):
    def func(log_prefix, path_to_csv, wandb_run_dir, **kwargs):
        if not os.path.exists(path_to_csv):
            wandb_entity = config.get('wandb_entity')
            wandb_proj = config.get('wandb_project')
            assert wandb_entity is not None and wandb_proj is not None, \
                "In order to connect with the WandB API, the wandb_entity and wandb_project " \
                "must be provided in the yaml!"
            run_id = run_id_from_wandb_run_dir(wandb_run_dir)
            run_api_identifier = f"{wandb_entity}/{wandb_proj}/{run_id}"
            print(log_prefix + "Downloading metrics from WandB")
            data = download_from_wandb(run_identifier=run_api_identifier)
            pd.DataFrame.from_dict(data).to_csv(path_to_csv)
        else:
            print(log_prefix + "Metrics are already downloaded")
    apply_func_to_metrics(exp_dirs, config, func)


def get_possible_key_paths(d):
    if not isinstance(d, dict):
        return []
    key_paths = []
    for k in d.keys():
        sub_keypaths = get_possible_key_paths(d.get(k))
        if sub_keypaths:
            [key_paths.append([k] + sk) for sk in sub_keypaths]
        else:
            key_paths.append([k])
    return key_paths


def collect_metrics_columns(exp_dirs: List[str], config: Dict):
    columns_seen = set()
    grouping_keys = {}

    def func(log_prefix, path_to_csv, experiment_config, columns_seen, grouping_keys: Dict, **kwargs):
        key_paths = get_possible_key_paths(experiment_config)
        key_paths = [tuple(k) for k in key_paths]
        for k in key_paths:
            if k not in grouping_keys.keys():
                grouping_keys[k] = 1
            else:
                grouping_keys[k] += 1
        try:
            data = pd.read_csv(path_to_csv)
            columns_seen |= set(data.columns)
            experiment_config['data'] = data
        except pd.errors.EmptyDataError:
            print(log_prefix + f"{config.get('csv_file_name')} was empty. Skipping.")
    apply_func_to_metrics(exp_dirs, config, func, columns_seen=columns_seen, grouping_keys=grouping_keys)
    return columns_seen, grouping_keys


def populate_group(exp_dirs: List[str], config: Dict, group: Group):
    def func(log_prefix, path_to_csv, experiment_config, group, **kwargs):
        try:
            data = pd.read_csv(path_to_csv)
            experiment_config['data'] = data
            err = group.add_run(experiment_config)
            if err is not None:
                print(log_prefix + err)
        except pd.errors.EmptyDataError:
            print(log_prefix + f"{config.get('csv_file_name')} was empty. Skipping.")
        except KeyError as e:
            print(log_prefix + f"Could not find keys {e.args[0]} in config.yaml. Skipping.")
    apply_func_to_metrics(exp_dirs, config, func, group=group)
    return group


def download_from_wandb(run_identifier: str) -> Dict:
    """

    :param run_identifier: Of the form 'entity/project/run_id'
    :return:
    """
    api = wandb.Api(timeout=30)
    run = api.run(run_identifier)
    scan = run.scan_history(page_size=500)
    metrics = {}

    nr_logged_metrics = 0
    for row in tqdm.tqdm(scan, total=scan.max_step):
        metrics_keys = set(metrics.keys())
        row_keys = set(row.keys())
        missing_keys_in_row = metrics_keys - row_keys
        for k in missing_keys_in_row:
            row[k] = float('NaN')
        new_keys = row_keys - metrics_keys
        for k in new_keys:
            metrics[k] = [float('NaN')] * nr_logged_metrics
        metrics_keys = set(metrics.keys())
        for k in metrics_keys:
            metrics[k].append(row[k])
        nr_logged_metrics += 1
        if not all([len(m) == nr_logged_metrics for m in metrics.values()]):
            print(1)
    return metrics


def build_group_from_config(d: Dict):
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
            child_group=build_group_from_config(d.get('child_group')),
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
            keys_to_plot=[re.compile(p) for p in d.get('keys_to_plot')],
            dtype=data_type,
            step_delta=d.get('step_delta')
        )


def build_plot_defs_from_config(d: Dict):
    global_fig_opt: dict = d.get('global_figure_options')
    plot_defs = []
    for title, plot_def in d.get('plot_definitions').items():
        fig_opt = global_fig_opt.copy()
        if (local_fig_opt := plot_def.get('figure_options')) is not None:
            fig_opt.update(local_fig_opt)
        assert 'content' in plot_def.keys()
        contents = []
        for content_type, content_kwargs in plot_def.get('content').items():
            if content_type == 'fill':
                content_kwargs['key_regexp'] = [re.compile(p) for p in content_kwargs.get('key_regexp')]
                contents.append(FillDef(**content_kwargs))
            elif content_type == 'line':
                content_kwargs['key_regexp'] = re.compile(content_kwargs.get('key_regexp'))
                contents.append(LineDef(**content_kwargs))
            else:
                raise NotImplementedError(f"Don't know what to do with a content type of {content_type}")
        plot_def.pop('content')
        plot_defs.append(PlotDef(line_defs=contents, title=title, figure_options=fig_opt))
    return plot_defs


