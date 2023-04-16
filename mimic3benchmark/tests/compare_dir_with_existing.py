from tqdm import tqdm

import os
import argparse
import pandas as pd


def formatter(x):
    try:
        x = float(x)
        return '{:.1f}'.format(x)
    except:
        return x


def get_all_csv_files(dir):
    csv_files = []
    for subdir, _, files in tqdm(os.walk(dir), desc='Getting all csv files'):
        for f in files:
            extension = f.split('.')[-1]
            if extension == 'csv':
                csv_files.append(os.path.join(os.path.relpath(subdir, dir),
                                              f))
    return set(csv_files)


def load_df(path):
    df = pd.read_csv(path, index_col=False)

    if 'AGE' in df.columns:
        df = df.drop('AGE', axis=1)
    if 'Age' in df.columns:
        df = df.drop('Age', axis=1)

    # convert all numbers to floats with fixed precision
    for col in df.columns:
        df[col] = df[col].apply(formatter)

    # sort by the first column that has unique values
    for col in df.columns:
        if len(df[col].unique()) == len(df):
            df = df.sort_values(by=col).reset_index(drop=True)
            break

    return df


def main():
    parser = argparse.ArgumentParser(description='Recursively checks whether all csv files are the same')
    parser.add_argument('--old_directory', type=str, required=True)
    parser.add_argument('--new_directory', type=str, required=True)
    args = parser.parse_args()
    print(args)

    # check that old and new directories contain the same csv files
    old_csv_files = get_all_csv_files(args.old_directory)
    new_csv_files = get_all_csv_files(args.new_directory)

    for s in old_csv_files:
        if s not in new_csv_files:
            print(f'{s} is missing in the new directory')

    for s in new_csv_files:
        if s not in old_csv_files:
            print(f'{s} appears in the new directroy but not in the old one')

    # iterate over all old csv files
    for csv_path in tqdm(old_csv_files, desc='checking csv files'):
        old_full_path = os.path.join(args.old_directory, csv_path)
        new_full_path = os.path.join(args.new_directory, csv_path)

        old_df = load_df(old_full_path)
        new_df = load_df(new_full_path)

        if not new_df.equals(old_df):
            print(f'differences found for {csv_path}')


if __name__ == "__main__":
    main()
