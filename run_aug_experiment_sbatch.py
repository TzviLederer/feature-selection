import os
import sys
import pandas as pd
from experiments_settings import DATASETS_FILES, OVERRIDE_LOGS
from run_aug_experiment import run_experiment


def run_all(results_file_name, logs_dir='logs_aug', overwrite_logs=False):
    os.makedirs(logs_dir, exist_ok=True)
    if len(sys.argv) == 1:
        datasets = list(pd.read_csv(results_file_name).dataset.unique())
        datasets_files = [name for arg in datasets for name in DATASETS_FILES if arg in name]
    else:
        datasets_files = [name for arg in sys.argv[1:] for name in DATASETS_FILES if arg in name]
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    dataset_file = datasets_files[task_id]
    print(f'Start Experiment, Dataset: {dataset_file}')
    output_log_file = run_experiment(dataset_file, results_file_name, logs_dir=logs_dir,
                                     overwrite_logs=overwrite_logs)
    print(f'Finished Experiment, Log file: {output_log_file}')


if __name__ == '__main__':
    run_all('unified_df.csv', overwrite_logs=OVERRIDE_LOGS)
