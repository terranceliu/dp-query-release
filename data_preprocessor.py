import numpy as np
import pandas as pd
from collections.abc import Iterable
from sklearn.preprocessing import LabelEncoder

class DataPreprocessingConfig():
    def __init__(self, config):
        self.attr_cat = config['attr_cat']
        self.attr_num = config['attr_num']
        self.mapping_cat_domain = config['mapping_cat_domain']
        self.mapping_num_bins = config['mapping_num_bins']

class DataPreprocessor():
    def __init__(self, config, default_num_bins=10):
        self.config = config
        self.default_num_bins = default_num_bins

        self.attr_cat = config.attr_cat
        self.attr_num = config.attr_num

        self.mapping_cat_domain = config.mapping_cat_domain
        self.mapping_num_bins = config.mapping_num_bins

        self.encoders = {}

    def _add_domain_rows(self, df):
        df = df.copy()
        if len(self.mapping_cat_domain) == 0:
            return df

        num_rows = max([len(x) for x in self.mapping_cat_domain.values()])
        df_extra = df.loc[np.arange(num_rows)].copy()
        for attr, categories in self.mapping_cat_domain.items():
            df_extra.loc[np.arange(len(categories)), attr] = categories
        df = pd.concat([df_extra, df]).reset_index(drop=True)
        return df

    def fit_cat(self, df):
        df = self._add_domain_rows(df)
        for attr in self.attr_cat:
            enc = LabelEncoder()
            enc.fit(df[attr].values)
            self.encoders[attr] = enc

    def transform_cat(self, df):
        for attr in self.attr_cat:
            enc = self.encoders[attr]
            encoded = enc.transform(df[attr].values)
            df.loc[:, attr] = encoded

    def inverse_transform_cat(self, df):
        for attr in self.attr_cat:
            enc = self.encoders[attr]
            df.loc[:, attr] = enc.inverse_transform(df[attr].values)

    def _get_bins(self, values, num_bins):
        bin_ranges = np.linspace(values.min(), values.max(), num_bins + 1)
        bin_ranges[0] = -np.inf
        bin_ranges[-1] = np.inf
        return bin_ranges.tolist()

    def fit_num(self, df):
        for attr in self.attr_num:
            bin_ranges = self._get_bins(df[attr].values, self.default_num_bins)
            if attr in self.mapping_num_bins.keys():
                if isinstance(self.mapping_num_bins[attr], list):
                    bin_ranges = self.mapping_num_bins[attr]
                    assert sorted(bin_ranges) == bin_ranges, '`bin_ranges` must be sorted.'
                elif isinstance(self.mapping_num_bins[attr], int):
                    num_bins = self.mapping_num_bins[attr]
                    bin_ranges = self._get_bins(df[attr].values, num_bins)
                else:
                    assert False, 'invalid config entry for {}'.format(attr)
            assert df[attr].min() >= bin_ranges[0], 'min value in bins is larger than the min value in the data'
            assert df[attr].max() >= bin_ranges[1], 'max value in bins is smaller than the max value in the data'

            self.mapping_num_bins[attr] = bin_ranges

    def transform_num(self, df):
        for attr, bin_ranges in self.mapping_num_bins.items():
            output = df[attr].copy()
            for i in range(len(bin_ranges) - 1):
                lower = float(bin_ranges[i])
                upper = float(bin_ranges[i + 1])
                if lower == upper:
                    mask = df[attr] == lower
                elif i > 0 and bin_ranges[i - 1] == lower:
                    mask = (df[attr] >= lower) & (df[attr] < upper)
                else:
                    mask = (df[attr] >= lower) & (df[attr] < upper)
                output[mask] = i
            df.loc[:, attr] = output

    def inverse_transform_num(self, df):
        for attr, bin_ranges in self.mapping_num_bins.items():
            output = df[attr].copy()
            for i in range(len(bin_ranges) - 1):
                lower = bin_ranges[i]
                upper = bin_ranges[i + 1]
                if lower == upper:
                    val = str(lower)
                elif i > 0 and bin_ranges[i - 1] == lower:
                    val = '({}, {})'.format(lower, upper)
                else:
                    val = '[{}, {})'.format(lower, upper)
                output[df[attr] == i] = val
            df.loc[:, attr] = output

    def fit(self, df):
        if isinstance(df, Iterable):
            df = pd.concat(df).reset_index(drop=True)
        self.fit_cat(df)
        self.fit_num(df)

    def transform(self, df):
        df = df.loc[:, self.attr_cat + self.attr_num].copy()
        self.transform_cat(df)
        self.transform_num(df)
        return df

    def inverse_transform(self, df):
        df = df.copy()
        self.inverse_transform_cat(df)
        self.inverse_transform_num(df)
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_domain(self):
        domain = {}
        for attr in self.attr_cat:
            domain[attr] = len(self.encoders[attr].classes_)
        for attr in self.attr_num:
            domain[attr] = len(self.mapping_num_bins[attr]) - 1
        return domain

def get_config(data_name):
    config = {}

    if data_name == 'adult':
        attr_cat = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country', 'income>50K']
        attr_num = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
        mapping_cat_domain = {}
        mapping_num_bins = {'age': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf],
                            'capital-gain': [0, 0, 1000, 2000, 3000, 4000,
                                             5000, 6000, 7000, 8000, 9000, 10000,
                                             15000, 20000, 30000, 50000, np.inf],
                            'capital-loss': [0, 0, 1000, 2000, 3000, 4000, np.inf],
                            'hours-per-week': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
                            }
    else:
        assert False, 'dataset config not found'

    config['attr_cat'] = attr_cat
    config['attr_num'] = attr_num
    config['mapping_cat_domain'] = mapping_cat_domain
    config['mapping_num_bins'] = mapping_num_bins

    return DataPreprocessingConfig(config)
