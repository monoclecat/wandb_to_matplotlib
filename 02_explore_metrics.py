from util import find_wandb_dirs, collect_metrics_columns, parse_config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = parse_config(args.config)
    exp_dirs = find_wandb_dirs(config.get('root_dir'))

    available_columns, grouping_keys = collect_metrics_columns(exp_dirs, config)
    print("Keys that can be grouped by: \n\t{}".format('\n\t'.join([f'{list(k)}\t- found in {v} experiments'
                                                                    for k, v in grouping_keys.items()])))
    print("Metrics that can be plotted: \n\t{}".format('\n\t'.join([k for k in available_columns])))
