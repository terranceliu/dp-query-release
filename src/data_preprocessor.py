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
    def __init__(self, config, fill_missing=False, default_num_bins=10):
        self.config = config
        self.fill_missing = fill_missing
        self.default_num_bins = default_num_bins

        self.attr_cat = config.attr_cat
        self.attr_num = config.attr_num

        self.mapping_cat_domain = config.mapping_cat_domain
        self.mapping_num_bins = config.mapping_num_bins

        self.encoders = {}

    def _get_df_domain(self, df):
        for attr in self.attr_cat:
            if attr in self.mapping_cat_domain.keys():
                if self.fill_missing:
                    self.mapping_cat_domain[attr].append('_OTHER')
            else:
                self.mapping_cat_domain[attr] = df[attr].unique().tolist()

        num_rows = max([len(x) for x in self.mapping_cat_domain.values()])
        df_domain = df.loc[:num_rows].copy()
        if len(df_domain) < num_rows:
            factor = np.ceil(num_rows / len(df_domain)).astype(int)
            df_domain = pd.concat([df_domain] * factor).reset_index(drop=True)
        for attr, categories in self.mapping_cat_domain.items():
            df_domain.loc[:len(categories) - 1, attr] = categories
            df_domain.loc[len(categories) + 1:, attr] = categories[0]
        return df_domain

    def fit_cat(self, df):
        df = self._get_df_domain(df)
        for attr in self.attr_cat:
            enc = LabelEncoder()
            enc.fit(df[attr].values)
            self.encoders[attr] = enc

    def transform_cat(self, df):
        for attr, categories in self.mapping_cat_domain.items():
            mask = ~df[attr].isin(categories)
            if mask.sum() > 0:
                if self.fill_missing and '_OTHER' in categories:
                    df.loc[mask, attr] = '_OTHER'
                else:
                    assert False, 'invalid value found in data (attr: {})'.format(attr)

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
            df.loc[:, attr] = output.astype(int)

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
            df = pd.concat(df)
        df.reset_index(drop=True, inplace=True)
        self.fit_cat(df)
        self.fit_num(df)

    def transform(self, df):
        if isinstance(df, Iterable):
            df = pd.concat(df)
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

"""
By default, the domain of each categorical variable is set to be the unique values found in the data.
Alternatively, one can explicitly pass in the domain values in `mapping_cat_domain`

Numerical/continuous attributes are preprocessed based on the dictionary `mapping_num_bins`
Usage:
    attr: num_bins (int)
        creates `num_bins` bins that are equally spaced across the min and max values for that attribute
    attr: bins (sorted list of ints)
        creates bins of the format
            bins=[a, b, c, d] -> bins = [a, b), [b, c), [c, d)
        if there is repeat (i.e., [a, a]), then we have
            bins=[a, a, b, c, d] -> [a, a], (a, b), [b, c), [c, d)
if an attribute in `attr_num` is missing from `mapping_num_bins`, it will default to
    attr: num_bins=10
"""
def get_config(data_name):
    config = {}

    if data_name == 'adult':
        attr_cat = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country', 'income>50K']
        attr_num = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
        mapping_cat_domain = {}
        mapping_num_bins = {'age': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf],
                            'capital-gain': [0, 0, 1000, 2000, 3000, 4000, 5000, 7500,
                                             10000, 15000, 20000, 30000, 50000, np.inf],
                            'capital-loss': [0, 0, 1000, 2000, 3000, 4000, np.inf],
                            'hours-per-week': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
                            }
    elif data_name == 'census':
        attr_num = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'dividends_from_stocks',
                    'weeks_worked_in_year']
        attr_cat = ['class_of_worker', 'detailed_industry_recode', 'detailed_occupation_recode', 'education',
                    'enroll_in_edu_inst_last_wk', 'marital_stat', 'major_industry_code', 'major_occupation_code',
                    'race', 'hispanic_origin', 'sex', 'member_of_a_labor_union', 'reason_for_unemployment',
                    'full_or_part_time_employment_stat', 'tax_filer_stat', 'region_of_previous_residence',
                    'state_of_previous_residence', 'detailed_household_and_family_stat',
                    'detailed_household_summary_in_household', 'migration_code-change_in_msa',
                    'migration_code-change_in_reg', 'migration_code-move_within_reg', 'live_in_this_house_1_year_ago',
                    'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer', 'family_members_under_18',
                    'country_of_birth_father',  'country_of_birth_mother', 'country_of_birth_self', 'citizenship',
                    'own_business_or_self_employed', 'fill_inc_questionnaire_for_veterans_admin', 'veterans_benefits',
                    'year', 'income']
        mapping_cat_domain = {}
        mapping_num_bins = {'age': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf],
                            'wage_per_hour': [0, 0, 100, 200, 300, 400, 500,
                                              600, 800, 1000, 1200, 1400, 1600,
                                              2000, 3000, 4000, 5000, np.inf],
                            'capital_gains': [0, 0, 1000, 2000, 3000, 4000, 5000, 7500,
                                              10000, 15000, 20000, 30000, 50000, np.inf],
                            'capital_losses': [0, 0, 1000, 2000, 3000, 4000, np.inf],
                            'dividends_from_stocks': [0, 0, 100, 200, 300, 500, 1000,
                                                      2500, 5000, 10000, 25000, 50000, np.inf],
                            'weeks_worked_in_year': [0, 0, 20, 40, 52, np.inf],
                            }
    else:
        assert False, 'dataset config not found'

    config['attr_cat'] = attr_cat
    config['attr_num'] = attr_num
    config['mapping_cat_domain'] = mapping_cat_domain
    config['mapping_num_bins'] = mapping_num_bins

    return DataPreprocessingConfig(config)