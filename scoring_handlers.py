import time
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pr_auc import pr_auc


def get_scoring(cv, y):
    is_multi = (len(y.unique()) > 2)
    if isinstance(cv, StratifiedKFold):
        return lambda estimator, X, y_true: _scoring_skf(estimator, X, y_true, multi=is_multi)
    return _scoring_lo


def _scoring_lo(estimator, X, y_true):
    return {
        'classifier_fit_time': estimator['clf'].fit_time,
        'feature_selector_fit_time': estimator['fs'].fit_time,
        **extract_selected_features(estimator),
        **estimator['clf'].metrics
    }


def _scoring_skf(estimator, X, y_true, multi=False):
    start = time.time()
    y_pred_proba = estimator.predict_proba(X)
    pred_time = time.time() - start
    return {
        'classifier_fit_time': estimator['clf'].fit_time,
        'feature_selector_fit_time': estimator['fs'].fit_time,
        'mean_inference_time': pred_time / X.shape[0],
        **extract_selected_features(estimator),
        **calculate_metrics(y_true, y_pred_proba, multi=multi)
    }


def calculate_metrics(y_true, y_pred_proba, multi=False):
    return {'acc': accuracy_score(y_true, y_pred_proba.argmax(axis=1)),
            'mcc': matthews_corrcoef(y_true, y_pred_proba.argmax(axis=1)),
            'roc_auc': roc_auc_score(y_true, y_pred_proba, average='weighted',
                                     multi_class='ovr') if multi else roc_auc_score(y_true, y_pred_proba[:, 1]),
            'pr_auc': pr_auc(y_true, y_pred_proba)}


def extract_selected_features(estimator):
    fs_input_features = estimator['dp'].get_feature_names_out()
    fs_scores = estimator['fs'].scores_
    if fs_scores.sum() == 0: # select fdr sometimes do not choose any feature, this is fix for logging only
        fs_scores += 1
    clf_input_features = estimator['fs'].get_feature_names_out(fs_input_features)
    res_scores = {f: s for f, s in zip(fs_input_features, fs_scores) if f in clf_input_features}
    return {f'{f}_feature_prob': res_scores.get(f, 0) for f in estimator['dp'].feature_names_in_}
