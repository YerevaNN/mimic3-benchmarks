import os
import shutil
import argparse


parser = argparse.ArgumentParser(description='Split data into train and test sets.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
args, _ = parser.parse_known_args()

testset = set()
with open("resources/testset.csv", "r") as test_set_file:
    for line in test_set_file:
        x, y = line.split(',')
        if int(y) == 1:
            testset.add(x)

def move_to_partition(patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        shutil.move(src, dest)


folders = os.listdir(args.subjects_root_path)
folders = list((filter(str.isdigit, folders)))
train_patients = [x for x in folders if not x in testset]
test_patients = [x for x in folders if x in testset]

assert len(set(train_patients) & set(test_patients)) == 0

move_to_partition(train_patients, "train")
move_to_partition(test_patients, "test")
