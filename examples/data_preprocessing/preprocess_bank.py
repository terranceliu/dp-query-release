import json
import pandas as pd
from preprocess_data import discretize_columns

if __name__ == "__main__":
    path = "./datasets/raw/bank-full.csv"
    df = pd.read_csv(path, sep=";")

    cols_categorical = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]
    cols_continuous = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    options = ["binsize", "custom", "binsize", "custom", "nbins", "nbins", "nbins"]
    params = [{"bin_size": 5},
              {"bin_ranges": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
              None,
              {"bin_ranges": [500, 1000, 2500]},
              {"nbins": 5, "minimum": 0, "maximum": 100},
              {"nbins": 20},
              None
              ]

    df_processed, domain = discretize_columns(df, cols_categorical, cols_continuous, cont_options=options, cont_params=params)

    csv_path = "datasets/bank.csv"
    domain_path = "datasets/bank-domain.json"

    # save
    df_processed.to_csv(csv_path, index=False)
    with open(domain_path, 'w') as f:
        json.dump(domain, f)