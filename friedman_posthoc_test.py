import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt


def warning_incomplete_experiments(df, algorithm_col, experiment_index_cols, metric_col):
    gb = df.groupby([algorithm_col, *experiment_index_cols])[metric_col].mean()
    warn_exps = gb[gb.isna()].reset_index().groupby('dataset')['filtering_algorithm'].apply(list).reset_index()
    for _, r in warn_exps.iterrows():
        print(f'Warning: Ignored Dataset "{r.dataset}" because its missing "{metric_col}" metric for {r.filtering_algorithm}')
    return warn_exps.dataset


def friedman_posthoc_test(results_csv_path, alpha=0.05, metric_col='test_roc_auc', algorithm_col='filtering_algorithm', experiment_index_cols=('dataset',)):
    df = pd.read_csv(results_csv_path)
    df = df[df.filtering_algorithm.isin(['fdr', 'mrmr', 'reliefF', 'rfe_svm'])] # TODO: Fix incomplete experiments
    warn_datasets = warning_incomplete_experiments(df, algorithm_col, experiment_index_cols, metric_col)
    df = df[~df.dataset.isin(warn_datasets)]

    mean_df = df.groupby([algorithm_col, *experiment_index_cols])[metric_col].mean().reset_index()
    metrics_df = mean_df.sort_values([algorithm_col, *experiment_index_cols]).groupby([algorithm_col])[metric_col].apply(list).reset_index()
    metrics_df[metric_col] = metrics_df[metric_col].map(lambda x: x[:10]) # TODO: Fix incomplete experiments
    data = metrics_df[metric_col].to_list()
    _, p_value = ss.friedmanchisquare(*data)
    if p_value < alpha:
        print(f'rejected null hypothesis')
        posthoc_res = sp.posthoc_nemenyi_friedman(np.array(data).T)
        r = np.argwhere(posthoc_res.to_numpy() < alpha)
        groups = metrics_df[algorithm_col].values
        metrics_means = mean_df.groupby([algorithm_col])[metric_col].mean()
        for x, y in groups[r]:
            if metrics_means[x] > metrics_means[y]:
                print(f'algorithm {x} is significantly better than algorithm {y} in terms of {metric_col}')
        sns.heatmap(posthoc_res, xticklabels=groups, yticklabels=groups, annot=True)
        plt.show()
    else:
        print(f'could not reject null hypothesis')


if __name__ == '__main__':
    friedman_posthoc_test('unified_df.csv')


