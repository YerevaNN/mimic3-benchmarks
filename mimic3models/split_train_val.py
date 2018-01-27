import os
import shutil
import argparse


parser = argparse.ArgumentParser(description="Split train data into train and validation sets.")
parser.add_argument('task', type=str, help="Possible values are: decompensation, "\
                    "in-hospital-mortality, length-of-stay, phenotyping, multitask  .")
args, _ = parser.parse_known_args()
assert args.task in ['decompensation', 'in-hospital-mortality', 'length-of-stay',
                     'phenotyping', 'multitask']

val_patients = set()
with open("mimic3models/valset.csv", "r") as valset_file:
    for line in valset_file:
        x, y = line.split(',')
        if int(y) == 1:
            val_patients.add(x)

with open("data/%s/train/listfile.csv" % args.task) as listfile:
    lines = listfile.readlines()
    header = lines[0]
    lines = lines[1:]

train_lines = [x for x in lines if x[:x.find("_")] not in val_patients]
val_lines = [x for x in lines if x[:x.find("_")] in val_patients]
assert len(train_lines) + len(val_lines) == len(lines)


with open("data/%s/train_listfile.csv" % args.task, "w") as train_listfile:
    train_listfile.write(header)
    for line in train_lines:
        train_listfile.write(line)

with open("data/%s/val_listfile.csv" % args.task, "w") as val_listfile:
    val_listfile.write(header)
    for line in val_lines:
        val_listfile.write(line)

shutil.copy("data/%s/test/listfile.csv" % args.task,
            "data/%s/test_listfile.csv" % args.task)
