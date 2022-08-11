import itertools
import numpy as np
import pandas as pd

from typing import *
from dataclasses import dataclass

@dataclass
class DataSchema:
    """
    Represents the column domains for a categorical data matrix. Each column has
    an index, a name, a list of allowed values, and a dtype.
    """

    column_names: List[str]
    column_values: List[List[Any]]
    column_dtypes: List[Any]
    name_to_ix: Dict[str, int]

    @staticmethod
    def infer_schema(df: pd.DataFrame):
        """`infer_schema(df: pd.DataFrame)`

        Return a `DataSchema` where the list of allowed values in each column is
        the set of unique values present in that column. The name and dtype are
        taken from pandas.
        """
        schema = DataSchema()
        for col_name in df.columns:
            unique_vals = set(df[col_name])
            schema.add_column(values=unique_vals, name=col_name, dtype=df[col_name].dtype)
        return schema

    def __init__(self):
        self.column_names = []
        self.column_values = []
        self.column_dtypes = []
        self.name_to_ix = {}

    def add_column(self, values: Iterable[Any], name=None, dtype=np.dtype("O")) -> None:
        """`add_column(values, name, dtype)`

        Adds a new column to `self` with the given list of allowed values, name,
        and dtype. If the name is ommitted, it will be set to `Column #`, where
        # is the index of the column. If the dtype is ommitted, it will be set
        to `np.dtype("O")`."""
        col_ix = len(self.column_names)
        if name is None:
            name = f"Column {col_ix}"
        if not isinstance(name, str):
            name = str(name)
        self.column_names.append(name)
        self.column_values.append(list(values))
        self.column_dtypes.append(dtype)
        self.name_to_ix[name] = col_ix
        None

    def column_index_from_name(self, name: str) -> int:
        """`column_index_from_name(name: str)`
        
        Given a column name, return the index of that column.
        """
        return self.name_to_ix[name]

    def num_columns(self) -> int:
        "`num_columns()` Return the number of columns in the data schema."
        return len(self.column_names)

    def get_column_values(self, column: Union[str, int]) -> List[Any]:
        """`get_column_values(column: Union[str, int])`

        Return the list of allowed values for the specified column. `column` may
        either be the column name, or the index of the column.
        """
        if isinstance(column, str):
            column = self.column_index_from_name(column)
        return self.column_values[column]

    def row_space_size(self) -> int:
        "`row_space_size()` Return the number of rows in the row space of this schema."
        return np.prod([len(cvs) for cvs in self.column_values])

    def row_space(self) -> pd.DataFrame:
        "`row_space()` Return a pandas DataFrame containing all possible rows for this schema."
        rows = pd.DataFrame.from_records(
            itertools.product(*self.column_values), columns=self.column_names,
        )
        for (colname, coltype) in zip(self.column_names, self.column_dtypes):
            rows[colname] = rows[colname].astype(coltype)
        return rows

    def __str__(self):
        result = "Data Schema:"
        result += f"\n  Row space size = {self.row_space_size()}"
        result += f"\n  {self.num_columns()} columns:"
        for i in range(self.num_columns()):
            name = self.column_names[i]
            dtype = self.column_dtypes[i]
            values = self.column_values[i]
            result += (
                f"\n    Col {i} ({name}): # values = {len(values)} (dtype = {dtype})"
            )

        return result