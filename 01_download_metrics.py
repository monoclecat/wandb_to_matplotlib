from util import find_wandb_dirs, download_metrics_and_save_as_csv, parse_config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = parse_config(args.config)
    exp_dirs = find_wandb_dirs(config.get('root_dir'))

    download_metrics_and_save_as_csv(exp_dirs, config)
