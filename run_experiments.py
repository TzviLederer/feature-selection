import os
import sys
from itertools import product
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from data_formatting import LABEL_COL
from data_preprocessor import build_data_preprocessor
from experiments_settings import DATASETS_FILES, FEATURES_SELECTORS, MODELS, KS, get_cv, get_scoring_metrics, N_JOBS, \
    get_scores_for_loo


def run_all(logs_dir='logs', overwrite_logs=False):
    os.makedirs(logs_dir, exist_ok=True)
    if len(sys.argv) == 1:
        datasets_files = DATASETS_FILES
    else:
        datasets_files = [name for arg in sys.argv[1:] for name in DATASETS_FILES if arg in name]

    for experiment_args in product(MODELS.keys(), datasets_files, FEATURES_SELECTORS.keys(), KS):
        print(f'Start Experiment, Settings: {experiment_args}')
        output_log_file = run_experiment(*experiment_args, logs_dir=logs_dir, overwrite_logs=overwrite_logs)
        print(f'Finished Experiment, Log file: {output_log_file}')


def run_experiment(estimator_name, filename, fs_name, k, logs_dir=None, overwrite_logs=True):
    log_filename = f'{"_".join([estimator_name, Path(filename).name, fs_name, str(k)])}.csv'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    # skip this experiment if exists
    if not overwrite_logs and Path(log_filename).exists():
        print('Exists, skipping')
        return log_filename

    df = pd.read_csv(filename)
    cv = get_cv(df)

    # check if the number of sample in each class is less than fold number
    df = drop_rare_labels(cv, df)

    X = df.drop(columns=[LABEL_COL])
    y = pd.Series(LabelEncoder().fit_transform(df[LABEL_COL]))

    cachedir1, cachedir2 = mkdtemp(), mkdtemp()
    pipeline = Pipeline(steps=[('preprocessing', build_data_preprocessor(X, memory=cachedir1)),
                               ('feature_selector', SelectKBest(FEATURES_SELECTORS[fs_name], k=k)),
                               ('classifier', MODELS[estimator_name])],
                        memory=cachedir2)

    base_log = {
        'learning_algorithm': estimator_name,
        'dataset': Path(filename).name,
        'filtering_algorithm': fs_name,
        'n_selected_features': k,
        'cv_method': str(cv),
        'n_samples': df.shape[0],
        'n_features_org': df.shape[1],
    }

    # If the cross validation is not leave one out or leave two out, we compute the results per fold
    if isinstance(cv, StratifiedKFold):
        results = cross_validate(pipeline, X, y, cv=cv, scoring=get_scoring_metrics(y), return_estimator=True,
                                 verbose=2, n_jobs=N_JOBS)
        pd.DataFrame(build_cv_logs(results, base_log)).to_csv(log_filename)

    else:
        # preprocess for feature selection
        X, selected_features_names, selected_features_scores = select_features(pipeline, X, y)

        # model pipeline
        results = fit_and_eval(estimator_name, X, y, cv, cachedir1, cachedir2)
        results.update({'selected_features_names': list(selected_features_names),
                        'selected_features_scores': selected_features_scores,
                        'fold': None})
        base_log.update(results)
        pd.DataFrame([base_log]).to_csv(log_filename)

    rmtree(cachedir1), rmtree(cachedir2)
    return log_filename


def fit_and_eval(estimator_name, X, y, cv, cachedir1, cachedir2):
    pipeline = Pipeline(steps=[('preprocessing', build_data_preprocessor(X, memory=cachedir1)),
                               ('classifier', MODELS[estimator_name])],
                        memory=cachedir2)
    outputs = cross_val_predict(pipeline, X, y, cv=cv, verbose=2, n_jobs=N_JOBS, method='predict_proba')
    return {k: scoring(y, outputs) for k, scoring in get_scores_for_loo(y).items()}


def select_features(pipeline, X, y):
    preprocessing = pipeline.steps[0][1]
    feature_selector = pipeline.steps[1][1]

    # apply feature selection
    X_pp = pd.DataFrame(preprocessing.fit_transform(X, y), index=X.index, columns=X.columns)
    feature_selector.fit(X_pp, y)
    selected_features_names = feature_selector.get_feature_names_out()
    indexes = list(map(lambda x: X.columns.to_list().index(x), selected_features_names))
    selected_features_scores = list(map(lambda x: feature_selector.scores_[x], indexes))

    X = pd.DataFrame(feature_selector.transform(X), columns=selected_features_names, index=X.index)
    return X, selected_features_names, selected_features_scores


def drop_rare_labels(cv, df):
    if isinstance(cv, StratifiedKFold):
        classes = df[LABEL_COL].value_counts() > cv.n_splits
        classes = classes[classes].index.to_list()
        df = df[df[LABEL_COL].apply(lambda x: x in classes)]
    return df


def build_cv_logs(results, base_log):
    logs = []
    for i in range(len(results['estimator'])):
        k_best, k_best_scores = extract_selected_features(results['estimator'][i])
        logs.append({
            **base_log,
            'fold': i + 1,
            'selected_features_names': ','.join([str(x) for x in k_best]),
            'selected_features_scores': ','.join(['%.4f' % x for x in k_best_scores]),
            **{k: '%.4f' % v[i] for k, v in results.items() if k != 'estimator'},
        })
    return logs


def extract_selected_features(estimator):
    fs_input_features = estimator['preprocessing'].get_feature_names_out()
    fs_scores = estimator['feature_selector'].scores_
    clf_input_features = estimator[:-1].get_feature_names_out()
    out_scores = {f: s for f, s in zip(fs_input_features, fs_scores) if f in clf_input_features}
    k_best, k_best_scores = zip(*sorted(out_scores.items(), key=lambda item: item[1], reverse=True))
    return k_best, k_best_scores


if __name__ == '__main__':
    run_all()
    # run_experiment('svm', 'data/preprocessed/ALLAML.csv', 'reliefF', 1)
