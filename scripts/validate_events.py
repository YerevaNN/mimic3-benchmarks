from __future__ import print_function

import os
import argparse
import pandas as pd
import numpy as np

def is_subject_folder(x):
    for c in x:
        if (c < '0' or c > '9'):
            return False
    return True


def main():
    #Set up some variables for verbose logging
    bad_pairs = set()
    missing_events = 0;

    n_events = 0
    emptyhadm = 0
    noicustay = 0
    recovered = 0
    couldnotrecover = 0
    icustaymissinginstays = 0
    nohadminstay = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('subjects_root_path', type=str,
                        help='Directory containing subject sub-directories.')
    args = parser.parse_args()
    print(args)

    subfolders = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subfolders))

    # get mapping for subject
    maps = {}
    for (index, subject) in enumerate(subjects):
        try:
            staysDF = pd.read_csv(os.path.join(args.subjects_root_path, subject, "stays.csv"), index_col=False)
            staysDF.columns = staysDF.columns.str.upper()
            staysDF.dropna(axis=0, how="any", subset=["HADM_ID"])
            if (index % 100 == 0):
                print("processed %d / %d" % (index+1, len(subjects)), "         \r")
            if os.path.isfile(os.path.join(args.subjects_root_path, subject, "events.csv")):
                eventsDF = pd.read_csv(os.path.join(args.subjects_root_path, subject, "events.csv"), index_col=False)
                n_events += eventsDF.shape[0]
                toProofread = eventsDF.merge(staysDF, left_on=["HADM_ID"], right_on=["HADM_ID"], how="left", suffixes=["", "_r"])

                #if icustayid is null but a record exists in staysDF, use that
                toProofread.loc[:, "ICUSTAY_ID"] = toProofread["ICUSTAY_ID"].fillna(toProofread["ICUSTAY_ID_r"])

                #gather stats
                emptyhadm += toProofread["HADM_ID"].isnull().sum()
                noicustay += toProofread["ICUSTAY_ID"].isnull().sum()
                couldnotrecover += (toProofread["ICUSTAY_ID"] != toProofread["ICUSTAY_ID_r"]).sum()
                icustaymissinginstays += toProofread["ICUSTAY_ID_r"].isnull().sum()

                #remove if ICUSTAY_ID isn't consistent in stays or events or if icustayid is missing entirely
                toProofread = toProofread[(toProofread["ICUSTAY_ID"] == toProofread["ICUSTAY_ID_r"])]
                toProofread = toProofread.dropna(axis=0, how="any", subset=["HADM_ID", "ICUSTAY_ID", "SUBJECT_ID"])

                recovered += toProofread.shape[0]
                toWrite = toProofread[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUEUOM"]] # remove any weird _l columns from merge
                toWrite.to_csv(os.path.join(args.subjects_root_path, subject, "events2.csv"))
            else:
                missing_events += 1
        except:
            print("error occurred, will go to next")
            couldnotrecover += 1

    #print bad_pairs
    print('n_events', n_events,
        'emptyhadm', emptyhadm,
        'noicustay', noicustay,
        'recovered', recovered ,
        'couldnotrecover', couldnotrecover ,
        'icustaymissinginstays', icustaymissinginstays ,
        'nohadminstay', nohadminstay,
        'noevents', missing_events )

if __name__=="__main__":
    main()
