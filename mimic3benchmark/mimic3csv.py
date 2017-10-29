import csv
import numpy as np
import os
import pandas as pd
import sys
import psycopg2
import json

from pandas import DataFrame

def convertListToSQL(listItems):
    '''
    Transform a list of items, (usually ids)
    from type int to format "(itemId1, itemId2)"
    for sql
    :param listItems a python list of stuff
    :return string in sql format for "WHERE var IN" would work
    '''
    toRet = ""
    for item in listItems:
        toRet += str(item) + ", "
    toRet = "(" + toRet[0:-2] + ")"
    return toRet

def query(sql, config):
    """
    :param sql Specific string query to run on the MIMIC3 sql database
    :param config a dict/object containing fields dbname, user, host, password, and port
        to create the connection to the database.
    :return: connection to database
    """
    try:
        config = Dict(config)
        conn = psycopg2.connect("dbname='" + str(config.dbname)
                                + "' user='" + str(config.user)
                                + "' host='" + str(config.host)
                                + "' password='" + str(config.password)
                                + "' port='" + str(config.port) + "' ")
    except:
        raise
    cur = conn.cursor()
    cur.execute("SET search_path TO mimiciii")
    return pd.read_sql(sql, conn)
def get_config(path):
    '''
    Gets the config to connect to database (stored in json file)
    :param path to the json
    :return an object with fields to key info about connection to database
    '''
    try:
        config = json.load(open(path, "r"))
    except:
        print "could not open path: " + path
    return config
def read_patients_table(mimic3_path, use_db = False):
    if (use_db):
        pats = query("SELECT * FROM patients", get_config(mimic3_path))
        pats.columns = pats.columns.str.upper()
    else:
        pats = DataFrame.from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats

def read_admissions_table(mimic3_path, use_db = False):
    if (use_db):
        admits = query("SELECT * FROM admissions", get_config(mimic3_path))
        admits.columns = admits.columns.str.upper()
    else:
        admits = DataFrame.from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits

def read_icustays_table(mimic3_path, use_db = False):
    if (use_db):
        stays = query("SELECT * FROM ICUSTAYS", get_config(mimic3_path))
        stays.columns = stays.columns.str.upper()
    else:
        stays = DataFrame.from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays

def read_icd_diagnoses_table(mimic3_path, use_db = False):
    if (use_db):
        codes = query("SELECT * FROM D_ICD_DIAGNOSES", get_config(mimic3_path))
        codes.columns = codes.columns.str.upper()
    else:
        codes = DataFrame.from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE','SHORT_TITLE','LONG_TITLE']]
    diagnoses = DataFrame.from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID','HADM_ID','SEQ_NUM']] = diagnoses[['SUBJECT_ID','HADM_ID','SEQ_NUM']].astype(int)
    return diagnoses

def read_events_table_by_row(mimic3_path, table, use_db = False, items_to_keep = None, subjects_to_keep = None):
    nb_rows = { 'chartevents': 263201376, 'labevents': 27872576, 'outputevents': 4349340 }
    if (use_db):
        events = query("SELECT * FROM table WHERE subject_id in " + convertListToSQL(subjects_to_keep) " AND itemid in " + convertListToSQL(items_to_keep), get_config(mimic3_path))
        events.columns = events.columns.str.upper()
        reader = events.iterrows()
    else:
        reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))
    for i,row in enumerate(reader):
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i, nb_rows[table.lower()]

def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[['ICD9_CODE','SHORT_TITLE','LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    codes['COUNT'] = diagnoses.groupby('ICD9_CODE')['ICUSTAY_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes.ix[codes.COUNT>0]
    if output_path:
        codes.to_csv(output_path, index_label='ICD9_CODE')
    return codes.sort_values('COUNT', ascending=False).reset_index()

def remove_icustays_with_transfers(stays):
    stays = stays.ix[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]

def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])

def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

def add_age_to_icustays(stays):
    stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    stays.AGE.ix[stays.AGE<0] = 90
    return stays

def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays

def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays

def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep.ix[(to_keep.ICUSTAY_ID>=min_nb_stays)&(to_keep.ICUSTAY_ID<=max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays

def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays.ix[(stays.AGE>=min_age)&(stays.AGE<=max_age)]
    return stays

def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

def break_up_stays_by_subject(stays, output_path, subjects=None, verbose=1):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays.ix[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')

def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None, verbose=1):
    subjects = diagnoses.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses.ix[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID','SEQ_NUM']).to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')

def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path, items_to_keep=None, subjects_to_keep=None, verbose=1, use_db = False):
    obs_header = [ 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM' ]
    if items_to_keep is not None:
        items_to_keep = set([ str(s) for s in items_to_keep ])
    if subjects_to_keep is not None:
        subjects_to_keep = set([ str(s) for s in subjects_to_keep ])

    class nonlocal: pass
    nonlocal.curr_subject_id = ''
    nonlocal.last_write_no = 0
    nonlocal.last_write_nb_rows = 0
    nonlocal.last_write_subject_id = ''
    nonlocal.curr_obs = []
    def write_current_observations():
        nonlocal.last_write_no += 1
        nonlocal.last_write_nb_rows = len(nonlocal.curr_obs)
        nonlocal.last_write_subject_id = nonlocal.curr_subject_id
        dn = os.path.join(output_path, str(nonlocal.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(nonlocal.curr_obs)
        nonlocal.curr_obs = []
    
    for row, row_no, nb_rows in read_events_table_by_row(mimic3_path, table, use_db=use_db, items_to_keep=items_to_keep, subjects_to_keep=subjects_to_keep):
        if verbose and (row_no % 100000 == 0):
            if nonlocal.last_write_no != '':
                sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...last write '
                                 '({3}) {4} rows for subject {5}'.format(table, row_no, nb_rows,
                                                                         nonlocal.last_write_no,
                                                                         nonlocal.last_write_nb_rows,
                                                                         nonlocal.last_write_subject_id))
            else:
                sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...'.format(table, row_no, nb_rows))
        
        if (subjects_to_keep is not None and row['SUBJECT_ID'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None and row['ITEMID'] not in items_to_keep):
            continue
        
        row_out = { 'SUBJECT_ID': row['SUBJECT_ID'],
                    'HADM_ID': row['HADM_ID'],
                    'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                    'CHARTTIME': row['CHARTTIME'],
                    'ITEMID': row['ITEMID'],
                    'VALUE': row['VALUE'],
                    'VALUEUOM': row['VALUEUOM'] }
        if nonlocal.curr_subject_id != '' and nonlocal.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
        nonlocal.curr_obs.append(row_out)
        nonlocal.curr_subject_id = row['SUBJECT_ID']
        
    if nonlocal.curr_subject_id != '':
        write_current_observations()

    if verbose and (row_no % 100000 == 0):
        sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...last write '
                         '({3}) {4} rows for subject {5}...DONE!\n'.format(table, row_no, nb_rows,
                                                                 nonlocal.last_write_no,
                                                                 nonlocal.last_write_nb_rows,
                                                                 nonlocal.last_write_subject_id))
