import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

BASELINES_NAMES = ['select_fdr_fs', 'mrmr_fs', 'reliefF_fs', 'rfe_svm_fs']
OUR_ALGORITHMS_NAMES = ['poly_svm_fs', 'poly_svm_fs_New', 'rbf_svm_fs', 'rbf_svm_fs_New', 'svm_fs', 'svm_fs_New','grey_wolf_fs' , 'grey_wolf_fs_New']

def get_common_experiments(df):
    cols = ['dataset', 'learning_algorithm', 'n_selected_features']
    gb = df.groupby(cols)['filtering_algorithm'].apply(list).apply(len)
    return df[df.set_index(cols).index.isin(gb[gb == gb.max()].index)]


def friedman_posthoc_test(results_csv_path, alpha=0.05, metric_col='test_roc_auc'):
    df = pd.read_csv(results_csv_path)
    df = get_common_experiments(df)
    experiment_index_cols = ['filtering_algorithm', 'dataset', 'learning_algorithm', 'n_selected_features']
    mean_df = df.groupby(experiment_index_cols).mean(metric_col).reset_index().sort_values(experiment_index_cols)
    metrics_df = mean_df.groupby(['filtering_algorithm'])[metric_col].apply(list).reset_index()
    assert len(set(metrics_df[metric_col].map(len))) == 1, 'Should have same # of values per filtering algorithm'

    data = metrics_df[metric_col].to_list()
    _, p_value = ss.friedmanchisquare(*data)
    if p_value < alpha:
        print(f'rejected null hypothesis')
        posthoc_res = sp.posthoc_nemenyi_friedman(np.array(data).T)
        r = np.argwhere(posthoc_res.to_numpy() < alpha)
        groups = metrics_df['filtering_algorithm'].values
        metrics_means = mean_df.groupby(['filtering_algorithm'])[metric_col].mean()

        res = pd.DataFrame([(x, y) for x, y in groups[r] if metrics_means[x] > metrics_means[y]], columns=['b', 'w'])
        signif_better = res.groupby('b')['w'].apply(list).reset_index().to_numpy()
        for x, y in signif_better:
            print(f'algorithm {x} is significantly better than algorithms {set(y)} in terms of {metric_col}')
        print('---------------------')
        for x, y in signif_better:
            if x in OUR_ALGORITHMS_NAMES:
                print(f'algorithm {x} is significantly better than baseline algorithms {set(y) & set(BASELINES_NAMES)} in terms of {metric_col}')
        plt.figure(figsize=(20, 15))
        sns.heatmap(posthoc_res, xticklabels=groups, yticklabels=groups, annot=True)
        plt.show()
    else:
        print(f'could not reject null hypothesis')


if __name__ == '__main__':
    friedman_posthoc_test('all_exp_df.csv')
