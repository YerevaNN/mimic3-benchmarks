from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import yaml
import random
random.seed(49297)
from tqdm import tqdm


def process_partition(args, definitions, code_to_group, id_to_group, group_to_id,
                      partition, eps=1e-6):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                cur_labels = [0 for i in range(len(id_to_group))]

                icustay = label_df['Icustay'].iloc[0]
                diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"),
                                           dtype={"ICD9_CODE": str})
                diagnoses_df = diagnoses_df[diagnoses_df.ICUSTAY_ID == icustay]
                for index, row in diagnoses_df.iterrows():
                    if row['USE_IN_BENCHMARK']:
                        code = row['ICD9_CODE']
                        group = code_to_group[code]
                        group_id = group_to_id[group]
                        cur_labels[group_id] = 1

                cur_labels = [x for (i, x) in enumerate(cur_labels)
                              if definitions[id_to_group[i]]['use_in_benchmark']]

                xty_triples.append((output_ts_filename, los, cur_labels))

    print("Number of created samples:", len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "train":
        xty_triples = sorted(xty_triples)

    codes_in_benchmark = [x for x in id_to_group
                          if definitions[x]['use_in_benchmark']]

    listfile_header = "stay,period_length," + ",".join(codes_in_benchmark)
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (x, t, y) in xty_triples:
            labels = ','.join(map(str, y))
            listfile.write('{},{:.6f},{}\n'.format(x, t, labels))


def main():
    parser = argparse.ArgumentParser(description="Create data for phenotype classification task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('--phenotype_definitions', '-p', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                        help='YAML file with phenotype definitions.')
    args, _ = parser.parse_known_args()

    with open(args.phenotype_definitions) as definitions_file:
        definitions = yaml.load(definitions_file)

    code_to_group = {}
    for group in definitions:
        codes = definitions[group]['codes']
        for code in codes:
            if code not in code_to_group:
                code_to_group[code] = group
            else:
                assert code_to_group[code] == group

    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, definitions, code_to_group, id_to_group, group_to_id, "test")
    process_partition(args, definitions, code_to_group, id_to_group, group_to_id, "train")


if __name__ == '__main__':
    main()
