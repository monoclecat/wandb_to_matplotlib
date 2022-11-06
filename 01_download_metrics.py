from util import find_wandb_dirs, download_metrics_and_save_as_csv, parse_config, download_run_configs, run_ids_in_exp_dirs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = parse_config(args.config)
    exp_dirs = find_wandb_dirs(config.get('root_dir'))
    existing_run_ids = run_ids_in_exp_dirs(exp_dirs)
    all_run_ids = download_run_configs(
        root_dir=config.get('root_dir'),
        wandb_entity=config.get('wandb_entity'),
        wandb_project=config.get('wandb_project'),
        exclude_run_ids=existing_run_ids,
    )
    exp_dirs = find_wandb_dirs(config.get('root_dir'))

    download_metrics_and_save_as_csv(exp_dirs, config)
