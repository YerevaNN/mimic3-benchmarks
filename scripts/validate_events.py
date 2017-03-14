import os
import argparse


def is_subject_folder(x):
    for c in x:
        if (c < '0' or c > '9'):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subjects_root_path', type=str,
                        help='Directory containing subject sub-directories.')
    args = parser.parse_args()
    print args
    
    subfolders = os.listdir(args.subjects_root_path)
    subjects = filter(is_subject_folder, subfolders)
    
    # get mapping for subject
    maps = {}
    for (index, subject) in enumerate(subjects):
        with open(os.path.join(args.subjects_root_path, subject, "stays.csv")) as f:
            rows = f.readlines()[1:]
            for row in rows:
                hadm_id = row.split(',')[1]
                icustay_id = row.split(',')[2]
                assert hadm_id != ""
                assert icustay_id != ""
                if (not subject in maps):
                    maps[subject] = dict()
                maps[subject][hadm_id] = icustay_id
    
    bad_pairs = set()
    
    n_events = 0
    emptyhadm = 0
    noicustay = 0
    recovered = 0
    couldnotrecover = 0
    icustaymissinginstays = 0
    nohadminstay = 0
    
    for (index, subject) in enumerate(subjects):
        new_lines = []
        with open(os.path.join(args.subjects_root_path, subject, "events.csv")) as f:
            lines = f.readlines()
            header = lines[0]
            lines = lines[1:]
            new_lines.append(header)
            for line in lines:
                mas = line.split(',')
                hadm_id = mas[1]
                icustay_id = mas[2]
                
                n_events += 1
                if (hadm_id == ""):
                    emptyhadm += 1
                    continue
                
                try:
                    icustay_id_from_maps = maps[subject][hadm_id]
                
                    if (icustay_id == ""):
                        noicustay += 1
                        try:
                            icustay_id = maps[subject][hadm_id]
                            recovered += 1
                            new_line = ','.join(mas[:2] + [icustay_id] + mas[3:])
                            new_lines.append(new_line)
                        except:
                            couldnotrecover += 1
                    else:
                        if icustay_id != icustay_id_from_maps:
                            icustaymissinginstays += 1
                        else:
                            new_line = ','.join(mas[:2] + [icustay_id] + mas[3:])
                            new_lines.append(new_line)
                except:
                    nohadminstay += 1
                    bad_pairs.add((subject, hadm_id))
                    
        
        with open(os.path.join(args.subjects_root_path, subject, "events.csv"), "w") as f:
            for new_line in new_lines:
                f.write(new_line)
        
        if (index % 100 == 0):
            print "processed %d / %d" % (index+1, len(subjects)), "         \r",
    print ""    
    
    #print bad_pairs
    print('n_events', n_events,
        'emptyhadm', emptyhadm,
        'noicustay', noicustay, 
        'recovered', recovered ,
        'couldnotrecover', couldnotrecover ,
        'icustaymissinginstays', icustaymissinginstays ,
        'nohadminstay', nohadminstay )
        
if __name__=="__main__":
    main()
