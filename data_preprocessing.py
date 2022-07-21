import re

import numpy as np
import pandas as pd
import scipy.io
from scipy.io import arff

LABEL_COL = 'y'
with_index = False

sk_list = [('data/scikit-feature/raw/ALLAML.mat', 'data/preprocessed/ALLAML.csv'),
           ('data/scikit-feature/raw/arcene.mat', 'data/preprocessed/arcene.csv'),
           ('data/scikit-feature/raw/BASEHOCK.mat', 'data/preprocessed/BASEHOCK.csv'),
           ('data/scikit-feature/raw/Carcinom.mat', 'data/preprocessed/Carcinom.csv'),
           ('data/scikit-feature/raw/CLL-SUB-111.mat', 'data/preprocessed/CLL-SUB-111.csv')]

arff_list = [('data/ARFF/raw/Breast.arff', 'data/preprocessed/Breast.csv'),
             ('data/ARFF/raw/CNS.arff', 'data/preprocessed/CNS.csv'),
             ('data/ARFF/raw/Lung.arff', 'data/preprocessed/Lung.csv'),
             ('data/ARFF/raw/Lymphoma.arff', 'data/preprocessed/Lymphoma.csv'),
             ('data/ARFF/raw/MLL.arff', 'data/preprocessed/MLL.csv')]

bioconductor_list = [('data/bioconductor/raw/ALL.csv', 'data/preprocessed/ALL.csv'),
                     ('data/bioconductor/raw/ayeastCC.csv', 'data/preprocessed/ayeastCC.csv'),
                     ('data/bioconductor/raw/bcellViper.csv', 'data/preprocessed/bcellViper.csv'),
                     ('data/bioconductor/raw/bladderbatch.csv', 'data/preprocessed/bladderbatch.csv'),
                     ('data/bioconductor/raw/breastCancerVDX.csv', 'data/preprocessed/breastCancerVDX.csv')]

microbiomic_list = [('data/microbiomic/raw/40168_2013_11_MOESM7_ESM/PBS.csv', 'data/preprocessed/PBS.csv'),
                    ('data/microbiomic/raw/40168_2013_11_MOESM3_ESM/CSS.csv', 'data/preprocessed/CSS.csv'),
                    ('data/microbiomic/raw/40168_2013_11_MOESM5_ESM/FSH.csv', 'data/preprocessed/FSH.csv'),
                    ('data/microbiomic/raw/40168_2013_11_MOESM1_ESM/CBH.csv', 'data/preprocessed/CBH.csv'),
                    ('data/microbiomic/raw/40168_2013_11_MOESM6_ESM/BP.csv', 'data/preprocessed/BP.csv'),
                    ('data/microbiomic/raw/40168_2013_11_MOESM4_ESM/FS.csv', 'data/preprocessed/FS.csv'),
                    ('data/microbiomic/raw/40168_2013_11_MOESM8_ESM/PDX.csv', 'data/preprocessed/PDX.csv'),
                    ('data/microbiomic/raw/40168_2013_11_MOESM2_ESM/CS.csv', 'data/preprocessed/CS.csv')]


def main():
    for input_path, output_path in microbiomic_list:
        print(input_path)
        preprocess_microbiomic(input_path, output_path)

    for input_path, output_path in bioconductor_list:
        print(input_path)
        preprocess_bioconductor(input_path, output_path)

    for input_path, output_path in arff_list:
        print(input_path)
        preprocess_arff(input_path, output_path)

    for input_path, output_path in sk_list:
        print(input_path)
        preprocess_sk(input_path=input_path, output_path=output_path)


def preprocess_sk(input_path, output_path):
    """
    Datasets link:
    https://drive.google.com/drive/folders/1gaMsh60L6ES3nm14azFvmg1CN7FdOX_D
    download and extract the directory
    """
    mat = scipy.io.loadmat(input_path)
    x, y = mat['X'], mat['Y']
    df = pd.DataFrame(x)
    df[LABEL_COL] = y
    df.to_csv(output_path, index=with_index)


def preprocess_microbiomic(input_path, output_path):
    df = pd.read_csv(input_path, index_col=0).T
    y = pd.Series(df.index, index=df.index).apply(lambda x: re.sub(r'\.\d+', '', x))
    df[LABEL_COL] = y
    df.to_csv(output_path, index=with_index)


def arrange_df(features, y_df):
    df = features[0]
    df = df.join(features[1])
    df = df.join(y_df)
    return df[~df.iloc[:, -1].isna()]


def txt2df(input_path_i):
    with open(input_path_i) as f:
        data = f.read()
    data = list(filter(lambda x: x, data.split('\n')))
    data = list(map(lambda x: x.split('\t'), data))
    data = np.array(data)
    df = pd.DataFrame(data[1:], columns=data[0])
    df.set_index(df.columns[0], inplace=True)
    return df


def preprocess_bioconductor(input_path, output_path):
    df = pd.read_csv(input_path, index_col=0, ).T
    df.rename({df.columns[0]: LABEL_COL}, axis=1, inplace=True)
    class_to_values = {v: i for i, v in enumerate(df[LABEL_COL].value_counts().index)}
    df[LABEL_COL] = df[LABEL_COL].map(class_to_values)
    reorder_cols = list(df.columns.delete(0)) + [LABEL_COL]
    df[reorder_cols].to_csv(output_path, index=with_index)


def preprocess_arff(input_path, output_path):
    """
    Dataset link:
    https://drive.google.com/drive/folders/1ak32sqSTlZ_3_GtJ_bMIvnAIPkCJ5Qbz?usp=sharing
    Download and extract the directory and the files
    """
    data = arff.loadarff(input_path)
    df = pd.DataFrame(data[0])
    df.rename({df.columns[-1]: LABEL_COL}, axis=1, inplace=True)
    class_to_values = {v: i for i, v in enumerate(df[LABEL_COL].value_counts().index)}
    df[LABEL_COL] = df[LABEL_COL].map(class_to_values)
    df.to_csv(output_path, index=with_index)


if __name__ == '__main__':
    main()
