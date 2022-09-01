import os
import csv
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--stateid', type=str)
parser.add_argument('--remove_existing', action='store_true')
args = parser.parse_args()
state_code = args.stateid

path = './datasets/raw/ppmf/2020-05-27-ppmf.csv'
save_dir = './datasets/raw/ppmf/by_state/'
save_path_base = os.path.join(save_dir, 'ppmf_{}.csv')
save_path = save_path_base.format(state_code)

if os.path.exists(save_path):
    if args.remove_existing:
        os.remove(save_path)
    else:
        assert False, 'file exists'

with open(path, 'r') as read_obj:
    reader = csv.reader(read_obj)

    header = next(reader)
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    state_idx = header.index('TABBLKST')

    with open(save_path, 'a') as f:
        for row in tqdm(reader):
            if row[state_idx] != state_code:
                continue

            writer = csv.writer(f)
            writer.writerow(row)