from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFdr
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
from utils import pr_auc

MODELS = {'nb': GaussianNB(),
          'svm': SVC(kernel='rbf'),
          'lr': LogisticRegression(max_iter=10_000),
          'rf': RandomForestClassifier(),
          'knn': KNeighborsClassifier()}

FEATURES_SELECTORS = {'fdr': SelectFdr(alpha=0.1)}

KS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

DATASETS_FILES = list(map(str, Path('data/preprocessed').glob('*.csv')))

METRICS = {'roc_auc': lambda y_true, y_score: roc_auc_score(y_true, y_score, multi_class='ovr'),
           'acc': accuracy_score,
           'mcc': matthews_corrcoef,
           'pr_auc': pr_auc}
