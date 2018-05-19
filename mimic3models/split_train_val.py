from __future__ import absolute_import
from __future__ import print_function

import shutil
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Split train data into train and validation sets.")
    parser.add_argument('dataset_dir', type=str, help='Path to the directory which contains the dataset')
    args, _ = parser.parse_known_args()

    val_patients = set()
    with open(os.path.join(os.path.dirname(__file__), 'resources/valset.csv'), 'r') as valset_file:
        for line in valset_file:
            x, y = line.split(',')
            if int(y) == 1:
                val_patients.add(x)

    with open(os.path.join(args.dataset_dir, 'train/listfile.csv')) as listfile:
        lines = listfile.readlines()
        header = lines[0]
        lines = lines[1:]

    train_lines = [x for x in lines if x[:x.find("_")] not in val_patients]
    val_lines = [x for x in lines if x[:x.find("_")] in val_patients]
    assert len(train_lines) + len(val_lines) == len(lines)

    with open(os.path.join(args.dataset_dir, 'train_listfile.csv'), 'w') as train_listfile:
        train_listfile.write(header)
        for line in train_lines:
            train_listfile.write(line)

    with open(os.path.join(args.dataset_dir, 'val_listfile.csv'), 'w') as val_listfile:
        val_listfile.write(header)
        for line in val_lines:
            val_listfile.write(line)

    shutil.copy(os.path.join(args.dataset_dir, 'test/listfile.csv'),
                os.path.join(args.dataset_dir, 'test_listfile.csv'))


if __name__ == '__main__':
    main()
