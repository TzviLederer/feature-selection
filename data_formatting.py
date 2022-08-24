import os
from pathlib import Path
import pandas as pd
import scipy.io
from scipy.io import arff

LABEL_COL = 'y'
with_index = False

# lists of path for the chosen datasets from each source:

sk_list = ['data/raw/scikit-feature/ALLAML.mat',
           'data/raw/scikit-feature/Carcinom.mat',
           'data/raw/scikit-feature/CLL-SUB-111.mat',
           'data/raw/scikit-feature/COIL20.mat']

arff_list = ['data/raw/ARFF/Breast.arff',
             'data/raw/ARFF/CNS.arff',
             'data/raw/ARFF/Lung.arff',
             'data/raw/ARFF/Lymphoma.arff',
             'data/raw/ARFF/MLL.arff']

bioconductor_list = ['data/raw/bioconductor/ALL.csv',
                     'data/raw/bioconductor/ayeastCC.csv',
                     'data/raw/bioconductor/bcellViper.csv',
                     'data/raw/bioconductor/bladderbatch.csv',
                     'data/raw/bioconductor/CLL.csv',
                     'data/raw/bioconductor/curatedOvarianData.csv',
                     ]

datamicroarray_list = ['data/raw/datamicroarray/borovecki_inputs.csv',
                       'data/raw/datamicroarray/christensen_inputs.csv',
                       'data/raw/datamicroarray/golub_inputs.csv',
                       'data/raw/datamicroarray/gravier_inputs.csv',
                       'data/raw/datamicroarray/khan_inputs.csv',
                       'data/raw/datamicroarray/pomeroy_inputs.csv',
                       'data/raw/datamicroarray/singh_inputs.csv',
                       'data/raw/datamicroarray/sorlie_inputs.csv',
                       'data/raw/datamicroarray/subramanian_inputs.csv',
                       'data/raw/datamicroarray/west_inputs.csv',
                       ]


def process_all(output_folder):
    # Transfers the input files listed above according to each source process function
    # and saves them at output_folder
    os.makedirs(output_folder, exist_ok=True)
    process_files(bioconductor_list, preprocess_bioconductor, output_folder),
    process_files(arff_list, preprocess_arff, output_folder),
    process_files(sk_list, preprocess_sk, output_folder)
    process_files(datamicroarray_list, preprocess_datamicroarray, output_folder)


def process_files(files, processor, output_folder):
    # process a single group of files from the same source using a give processor function
    for input_path in files:
        processor(input_path, build_output_path(input_path, output_folder))
        print(f'Finished processing {input_path}')


def build_output_path(input_path, output_folder):
    # build the path of the new file - same name of the original one but as csv format.
    return f'{output_folder}/{"".join(Path(input_path).name.split(".")[:-1])}.csv'


def preprocess_sk(input_path, output_path):
    # preprocessing for the scikit-feature source
    """
    Datasets link:
    https://drive.google.com/drive/folders/1gaMsh60L6ES3nm14azFvmg1CN7FdOX_D
    download and extract the directory
    """
    mat = scipy.io.loadmat(input_path)
    x, y = mat['X'], mat['Y']
    df = pd.DataFrame(x)
    df[LABEL_COL] = y

    class_to_values = {v: i for i, v in enumerate(df[LABEL_COL].value_counts().index)}
    df[LABEL_COL] = df[LABEL_COL].map(class_to_values)

    df.to_csv(output_path, index=with_index)


def preprocess_bioconductor(input_path, output_path):
    # preprocessing for the bioconductor source
    """
        Datasets link:
        https://drive.google.com/drive/folders/1gd-EZyTBFg-bs1MZ7yA6X2MhhuAV0scE?usp=sharing
        download and extract the directory
        """
    df = pd.read_csv(input_path, index_col=0, ).T
    df.rename({df.columns[0]: LABEL_COL}, axis=1, inplace=True)
    class_to_values = {v: i for i, v in enumerate(df[LABEL_COL].value_counts().index)}
    df[LABEL_COL] = df[LABEL_COL].map(class_to_values)
    reorder_cols = list(df.columns.delete(0)) + [LABEL_COL]
    df[reorder_cols].to_csv(output_path, index=with_index)


def preprocess_arff(input_path, output_path):
    # preprocessing for the arff source
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


def preprocess_datamicroarray(input_path, output_path):
    # preprocessing for the datamicroarray source
    """
        Datasets link:
        https://drive.google.com/drive/folders/1hEiPKWSJ-_ecOtDQVZXg7Z9bSd2vfe0W?usp=sharing
        download and extract the directory
        """
    df = pd.read_csv(input_path, header=None)
    df[LABEL_COL] = pd.read_csv(f'{input_path.split("_")[0]}_outputs.csv', header=None)[0]
    df.columns = [str(c) for c in df.columns]
    df.to_csv(output_path, index=with_index)


if __name__ == '__main__':
    process_all('data/preprocessed')
