from collections import defaultdict
import operator

import numpy as np
import pandas as pd

# you need to set these values
ANOM_COLS = []  # list of df columns to include in mondrian anonymisation
CAT_COLS = []  # list of df columns to convert from categories to numerical values
K = 3  # k-anonymization

RANGE = []  # array of range of values for each column in ANOM_COLS
RESULTS = []  # list to store array of partitioned dataframes returned by mondrian


class Partition(object):

    def __init__(self, data):
        self.df = data
        self.disallowed = []

    def __len__(self):
        return len(self.df)

    def get_normalized_width(self):
        width = self.df.max() - self.df.min()
        return width / np.array(RANGE)

    def choose_dimension(self):
        # largest amount of unique values first
        most_values = sorted([(n, len(col.unique())) for n, col in self.df.iteritems()],
                             key=operator.itemgetter(1), reverse=True)
        for x, _ in most_values:
            if x not in self.disallowed:
                return x
        return None

        # highest normalized width first
        # TODO: check what method makes more sense and what exactly the
        # normalized width is suppose to be

        # for x in np.argsort(self.get_normalized_width())[::-1].index.values:
        #     if x not in self.disallowed:
        #         return x
        # return None

    def get_split_value(self, dim):
        return self.df[dim].median()


def cat_to_num(df):
    """ Convert categorical values to numerical values from 0 to n.
        Return the mapped dataframe and the mapping dictionary.
    """
    category_maps = defaultdict(dict)
    for col in df.columns:
        unique_values = sorted(df[col].unique())
        if col in CAT_COLS:
            for i, v in enumerate(unique_values):
                category_maps[col][v] = i
            df[col] = df[col].apply(lambda x: category_maps[col][x])
    return df, category_maps


def num_to_cat(df, category_maps):
    for col in df.columns:
        if col in CAT_COLS:
            reversed_map = dict([(v, k) for k, v in category_maps[col].items()])
            df[col] = df[col].apply(lambda x: reversed_map[x])
    return df


def aggregate_partitions(partitions):
    df_list = []
    for res in partitions:
        for col in res.columns:
            res[col] = int(res[col].mean())
        df_list.append(res)
    return pd.concat(df_list)


def mondrian(partition):
    # min partition size stopping criterion
    if len(partition) <= (2 * K - 1):
        RESULTS.append(partition.df)
    else:
        dim = partition.choose_dimension()
        # stop if no allowable dimension to split on
        if dim is None:
            RESULTS.append(partition.df)
        else:
            split_val = partition.get_split_value(dim)
            # split partition
            if partition.df[dim].max == split_val:
                lhs = Partition(partition.df[partition.df[dim] < split_val])
                rhs = Partition(partition.df[partition.df[dim] >= split_val])
            else:
                lhs = Partition(partition.df[partition.df[dim] <= split_val])
                rhs = Partition(partition.df[partition.df[dim] > split_val])
            if not len(lhs) or not len(rhs):
                partition.disallowed.append(dim)
                mondrian(partition)
            else:
                mondrian(lhs)
                mondrian(rhs)


def run(df, anom_cols, cat_cols, k=5):
    global RANGE
    # select columns to anonymize
    df_anom = df[ANOM_COLS]
    # convert categorical values to numerical
    df_anom, category_maps = cat_to_num(df_anom)
    # compute range of values for each column
    RANGE = (df_anom.max() - df_anom.min()).values
    # create partition object
    partition = Partition(df_anom)
    # run mondrian
    mondrian(partition)
    # aggregate partitions
    anom = aggregate_partitions(RESULTS)
    # convert numericals back to categories
    anom = num_to_cat(anom, category_maps)
    # rejoin anonymized columns with original dataframe
    tmp = df[df.columns.difference(ANOM_COLS)]
    return anom.join(tmp)


def run_adults():
    """ Run Mondrian on the adults dataset. """
    global ANOM_COLS, CAT_COLS, K

    attributes = ['age', 'workclass', 'final_weight', 'education',
                  'education_num', 'marital_status', 'occupation', 'relationship',
                  'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                  'native_country', 'class']
    df = pd.read_csv('../data/adult.data', names=attributes)

    ANOM_COLS = ['age', 'education_num', 'race', 'sex']
    CAT_COLS = ['race', 'sex']
    K = 5

    df = run(df, ANOM_COLS, CAT_COLS, K)
    print(df.head())
    return df

if __name__ == '__main__':
    run_adults()
