import os
import argparse
import numpy as np
from datetime import datetime
import pandas as pd
import yaml
import random
random.seed(49297)


parser = argparse.ArgumentParser(description="Create data for multitask prediction.")
parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
parser.add_argument('--phenotype_definitions', '-p', type=unicode, default='resources/hcup_ccs_2015_definitions.yaml',
                    help='YAML file with phenotype definitions.')
args, _ = parser.parse_known_args()

with open(args.phenotype_definitions) as definitions_file:
    definitions = yaml.load(definitions_file)

code_to_group = {}
for group in definitions:
    codes = definitions[group]['codes']
    for code in codes:
        if (code not in code_to_group):
            code_to_group[code] = group
        else:
            assert code_to_group[code] == group
            
id_to_group = sorted(definitions.keys())
group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))


if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
    
    
def process_partition(partition, sample_rate=1.0, shortest_length=4,
                      eps=1e-6, future_time_interval=24.0, fixed_hours=48.0):
   
    output_dir = os.path.join(args.output_path, partition)
    if (not os.path.exists(output_dir)):
        os.mkdir(output_dir)
    
    filenames = []
    loses = []
    
    fixed_mort_masks = []
    fixed_mort_labels = []
    fixed_mort_positions = []
    
    los_masks = []
    los_labels = []
    
    phenotype_labels = []
    
    swat_masks = []
    swat_labels = []
    
    patients = filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition)))
    
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder))
        stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))
        
        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
                
                # empty label file, skip globally
                if (label_df.shape[0] == 0):
                    print "\n\t(empty label file)", patient, ts_filename
                    continue
                
                # find length of stay, skip globally if it is missing
                los = 24.0 * label_df.iloc[0]['Length of Stay'] # in hours
                if (pd.isnull(los)):
                    print "\n\t(length of stay is missing)", patient, ts_filename
                    continue
                
                # find all event in ICU, skip globally if there is no event in ICU
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                                if t > -eps and t < los - eps]
                event_times = [t for t in event_times
                                if t > -eps and t < los - eps]
                
                if (len(ts_lines) == 0):
                    print "\n\t(no events in ICU) ", patient, ts_filename
                    continue
                
                # add length of stay
                loses.append(los)
                
                # find in hospital mortality
                mortality = int(label_df.iloc[0]["Mortality"])
                
                # write episode data and add file name
                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)
                filenames.append(output_ts_filename)
                
                # create fixed mortality
                fm_label = mortality
                
                fm_mask = 1
                if los < fixed_hours - eps:
                    fm_mask = 0
                if event_times[0] > fixed_hours + eps:
                    fm_mask = 0
                
                fm_position = 47
                if fm_mask == 0:
                    fm_position = 0
                
                fixed_mort_masks.append(fm_mask)
                fixed_mort_labels.append(fm_label)
                fixed_mort_positions.append(fm_position)
                
                # create length of stay
                sample_times = np.arange(0.0, los - eps, sample_rate)
                sample_times = np.array([int(x+eps) for x in sample_times])
                cur_los_masks = map(int, (sample_times > shortest_length) & (sample_times > event_times[0]))
                cur_los_labels = los - sample_times
                
                los_masks.append(cur_los_masks)
                los_labels.append(cur_los_labels)
                
                # create phenotypes
                cur_phenotype_labels = [0 for i in range(len(id_to_group))]
                icustay = label_df['Icustay'].iloc[0]
                diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"), dtype={"ICD9_CODE": str})
                diagnoses_df = diagnoses_df[diagnoses_df.ICUSTAY_ID == icustay]
                
                for index, row in diagnoses_df.iterrows():
                    if row['USE_IN_BENCHMARK']:
                        code = row['ICD9_CODE']
                        group = code_to_group[code]
                        group_id = group_to_id[group]
                        cur_phenotype_labels[group_id] = 1
                
                cur_phenotype_labels = [x for (i, x) in enumerate(cur_phenotype_labels)
                                 if definitions[id_to_group[i]]['use_in_benchmark']]
                phenotype_labels.append(cur_phenotype_labels)
                
                # create swat
                stay = stays_df[stays_df.ICUSTAY_ID == icustay]
                deathtime = stay['DEATHTIME'].iloc[0]
                intime = stay['INTIME'].iloc[0]
                if (pd.isnull(deathtime)):
                    lived_time = 1e18
                else:
                    lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -\
                                  datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0
                
                sample_times = np.arange(0.0, los - eps, sample_rate)
                sample_times = np.array([int(x+eps) for x in sample_times])
                cur_swat_masks = map(int, (sample_times > shortest_length) & (sample_times > event_times[0]) &
                                          (sample_times < lived_time + eps))
                cur_swat_labels = [(mortality & int(lived_time - t < future_time_interval))
                                       for t in sample_times]
                swat_masks.append(cur_swat_masks)
                swat_labels.append(cur_swat_labels)
        
        if ((patient_index + 1) % 100 == 0):
            print "\rprocessed %d / %d patients" % (patient_index + 1, len(patients)),
    
    def permute(arr, p):
        return [arr[index] for index in p]
    
    if partition == "train":
        perm = range(len(filenames))
        random.shuffle(perm)
    if partition == "test":
        perm = list(np.argsort(filenames))
    
    filenames = permute(filenames, perm)
    loses = permute(loses, perm)
    
    fixed_mort_masks = permute(fixed_mort_masks, perm)
    fixed_mort_labels = permute(fixed_mort_labels, perm)
    fixed_mort_positions = permute(fixed_mort_positions, perm)
    
    los_masks = permute(los_masks, perm)
    los_labels = permute(los_labels, perm)
    
    phenotype_labels = permute(phenotype_labels, perm)
    
    swat_masks = permute(swat_masks, perm)
    swat_labels = permute(swat_labels, perm)
    
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        header = "filename, length of stay, fixed mortality task (pos;mask;label), "\
                 "length of stay task (masks;labels), phenotyping task (labels), "\
                 "swat (masks;labels)"
        listfile.write(header + "\n")
        
        for index in range(len(filenames)):
            fname = filenames[index]
            los = "%.6f" % loses[index]
            
            fm = "%d;%d;%d" % (fixed_mort_positions[index], fixed_mort_masks[index], fixed_mort_labels[index])
            
            ls1 = ";".join(map(str, los_masks[index]))
            ls2 = ";".join(map(lambda x: "%.6f" % x, los_labels[index]))
            
            ph = ";".join(map(str, phenotype_labels[index]))
            
            sw1 = ";".join(map(str, swat_masks[index]))
            sw2 = ";".join(map(str, swat_labels[index]))
            
            listfile.write(','.join([fname, los, fm, ls1 + ";" + ls2, ph, sw1 + ";" + sw2]) + "\n")


process_partition("test")
process_partition("train")
