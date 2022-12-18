import os
import argparse

from src.utils import get_data

'''
YEAR=2018
STATES=(CA NY TX FL PA)
TASKS=(income employment coverage mobility travel)

for STATE in "${STATES[@]}"
do
    for TASK in "${TASKS[@]}"
    do
        python examples/data_preprocessing/create_train_test_split.py --mixed \
        --dataset folktables_${YEAR}_${TASK}_${STATE}
    done
done
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--root', type=str, default='./datasets/')
    args = parser.parse_args()
    if args.mixed:
        args.dataset += '-mixed'
    return args

def split_data(df, frac=0.8, seed=0):
    df_split1 = df.sample(frac=frac, random_state=seed)
    df_split2 = df.drop(df_split1.index)
    return df_split1.reset_index(drop=True), df_split2.reset_index(drop=True)

args = get_args()

dataset = get_data(args.dataset, root_path=args.root)
df_train, df_test = split_data(dataset.df)

train_csv_path = os.path.join(args.root, args.dataset + '-train.csv')
test_csv_path = os.path.join(args.root, args.dataset + '-test.csv')
df_train.to_csv(train_csv_path, index=False)
df_test.to_csv(test_csv_path, index=False)

domain_path = os.path.join(args.root, 'domain/', args.dataset + '-domain.json')
train_domain_path = os.path.join(args.root, 'domain/', args.dataset + '-train-domain.json')
test_domain_path = os.path.join(args.root, 'domain/', args.dataset + '-test-domain.json')
os.symlink(os.path.realpath(domain_path), train_domain_path)
os.symlink(os.path.realpath(domain_path), test_domain_path)

if args.mixed:
    for suffix in ['-cat', '-num']:
        os.symlink(os.path.realpath(train_csv_path), train_csv_path.replace('-mixed', suffix))
        os.symlink(os.path.realpath(test_csv_path), test_csv_path.replace('-mixed', suffix))

        domain_path = os.path.join(args.root, 'domain/', args.dataset.replace('-mixed', suffix) + '-domain.json')
        os.symlink(os.path.realpath(domain_path), train_domain_path.replace('-mixed', suffix))
        os.symlink(os.path.realpath(domain_path), test_domain_path.replace('-mixed', suffix))



