from __future__ import absolute_import
from __future__ import print_function

from tqdm import tqdm

import pickle as pkl
import hashlib
import os
import argparse
import pandas as pd


def formatter(x):
    try:
        x = float(x)
        return '{:.1f}'.format(x)
    except:
        return x


def main():
    parser = argparse.ArgumentParser(description='Recursively produces hashes for all tables inside this directory')
    parser.add_argument('--directory', '-d', type=str, required=True, help='The directory to hash.')
    parser.add_argument('--output_file', '-o', type=str, default='hashes.pkl')
    args = parser.parse_args()
    print(args)

    # count the directories
    total = 0
    for subdir, dirs, files in tqdm(os.walk(args.directory), desc='Counting directories'):
        total += len(files)

    # change directory to args.directory
    initial_dir = os.getcwd()
    os.chdir(args.directory)

    # iterate over all subdirectories
    hashes = {}
    pbar = tqdm(total=total, desc='Iterating over files')
    for subdir, dirs, files in os.walk('.'):
        for file in files:
            pbar.update(1)
            # skip files that are not csv
            extension = file.split('.')[-1]
            if extension != 'csv':
                continue

            full_path = os.path.join(subdir, file)
            df = pd.read_csv(full_path, index_col=False)

            # convert all numbers to floats with fixed precision
            for col in df.columns:
                df[col] = df[col].apply(formatter)

            # sort by the first column that has unique values
            for col in df.columns:
                if len(df[col].unique()) == len(df):
                    df = df.sort_values(by=col).reset_index(drop=True)
                    break

            # convert the data frame to string and hash it
            df_str = df.to_string().encode()
            hashcode = hashlib.md5(df_str).hexdigest()
            hashes[full_path] = hashcode
    pbar.close()

    # go to the initial directory and save the results
    os.chdir(initial_dir)
    with open(args.output_file, 'wb') as f:
        pkl.dump(hashes, f)


if __name__ == "__main__":
    main()
