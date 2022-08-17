import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""
Discrete columns only
"""
def get_domain_rows(domain, discrete_columns):
    max_attr_domain = max(domain.shape)
    df_domain = pd.DataFrame(np.zeros((max_attr_domain, len(domain))), columns=domain.attrs, dtype=int)
    for col in discrete_columns:
        domain_vals = np.arange(domain[col])
        df_domain.loc[:domain[col] - 1, col] = domain_vals

    return df_domain

class DummyEncoderTransformer():
    def transform(self, x):
        return x

    def reverse_transform(self, x):
        return x


class DataTransformer(object):
    """Data Transformer.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """
    def __init__(self, domain):
        self.domain = domain

        self.output_info = []
        self.output_dimensions = 0
        self.meta = []
        self._fit()

    def _fit_discrete(self, column, values):
        encoder = OneHotEncoder()
        encoder.fit(values.reshape(-1, 1))
        categories = len(encoder.categories_[0])

        return {
            'name': column,
            'encoder': encoder,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories
        }

    def _fit(self):
        # self.dtypes = data.infer_objects().dtypes
        for attr, domain_size in self.domain.config.items():
            if domain_size > 0:
                meta = self._fit_discrete(attr, np.arange(domain_size))
            else:
                assert False, 'continuous not implemented'

            self.output_info += meta['output_info']
            self.output_dimensions += meta['output_dimensions']
            self.meta.append(meta)

    def _transform_discrete(self, column_meta, data):
        encoder = column_meta['encoder']
        return encoder.transform(data)

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self.meta:
            column_name = meta['name']
            column_data = data[[column_name]].values
            if self.domain[column_name] > 0:
                values.append(self._transform_discrete(meta, column_data))
            else:
                assert False, 'continuous not implemented'

        return np.concatenate(values, axis=1).astype(float)

    def _inverse_transform_discrete(self, column_meta, data):
        encoder = column_meta['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(self, data):
        start = 0
        output = []
        column_names = []
        for meta in self.meta:
            column_name = meta['name']
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]
            if self.domain[column_name] > 0:
                inverted = self._inverse_transform_discrete(meta, columns_data)
            else:
                assert False, 'continuous not implemented'

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names)

        return output