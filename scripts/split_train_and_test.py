import os
import shutil
import argparse
import random
random.seed(47297)


parser = argparse.ArgumentParser(description='Split data into train and test sets.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
args, _ = parser.parse_known_args()


def move_to_partition(patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        shutil.move(src, dest)        


folders = os.listdir(args.subjects_root_path)
folders = filter(str.isdigit, folders)
random.shuffle(folders)
train_cnt = int(0.85 * len(folders))

train_patients = sorted(folders[:train_cnt])
test_patients = sorted(folders[train_cnt:])
assert len(set(train_patients) & set(test_patients)) == 0

move_to_partition(train_patients, "train")
move_to_partition(test_patients, "test")
