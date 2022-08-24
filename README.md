# feature-selection
Final assignment of Computational Learning course

## Usage
Use the script `data_formatting.py` to convert data to unified format. \
Use the script `run_experiments.py` to execute all requested experiments over all datasets (for parts 2,3). \
Use the script `run_aug_experiment.py` to execute the requested augmentation applied experiments (for part 4). \
Use the script `friedman_posthoc_test.py` to execute the friedman and posthoc tests (for part 5). \
The scripts `run_experiments_sbatch.py` and `run_aug_experiment_sbatch.py` were used to \
execute the experiments in parallel on BGU's CPU cluster (using sbatch arrays). \

## Notebooks

`illustrations.ipynb` - used to execute the toy example illustrations. \
`analyze_results.ipynb` - used to analyze the experiments results.