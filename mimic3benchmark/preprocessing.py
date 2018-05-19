from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re

from pandas import DataFrame, Series

from mimic3benchmark.util import *

###############################
# Non-time series preprocessing
###############################

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}


e_map = {'ASIAN': 1,
         'BLACK': 2,
         'HISPANIC': 3,
         'WHITE': 4,
         'OTHER': 5,  # map everything else to 5 (OTHER)
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         '': 0}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}


def assemble_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.ICUSTAY_ID, 'Age': stays.AGE, 'Length of Stay': stays.LOS,
            'Mortality': stays.MORTALITY}
    data.update(transform_gender(stays.GENDER))
    data.update(transform_ethnicity(stays.ETHNICITY))
    data['Height'] = np.nan
    data['Weight'] = np.nan
    data = DataFrame(data).set_index('Icustay')
    data = data[['Ethnicity', 'Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)


diagnosis_labels = ['4019', '4280', '41401', '42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720',
                    '2859', '2449', '486', '2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070',
                    '40390', '3051', '412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867',
                    '4241', '40391', '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749',
                    '4168', '5180', '45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111',
                    'V1251', '30000', '3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652',
                    '99811', '431', '28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832',
                    'E8788', '00845', '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703',
                    '2809', '5712', '27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254',
                    '42823', '28529', 'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859',
                    'V667', 'E8497', '79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770',
                    'V5865', '99662', '28860', '36201', '56210']


def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    diagnoses['VALUE'] = 1
    labels = diagnoses[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates().pivot(index='ICUSTAY_ID', columns='ICD9_CODE', values='VALUE').fillna(0).astype(int)
    for l in diagnosis_labels:
        if l not in labels:
            labels[l] = 0
    labels = labels[diagnosis_labels]
    return labels.rename_axis(dict(zip(diagnosis_labels, ['Diagnosis ' + d for d in diagnosis_labels])), axis=1)


def add_hcup_ccs_2015_groups(diagnoses, definitions):
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])
    diagnoses['HCUP_CCS_2015'] = diagnoses.ICD9_CODE.apply(lambda c: def_map[c][0] if c in def_map else None)
    diagnoses['USE_IN_BENCHMARK'] = diagnoses.ICD9_CODE.apply(lambda c: int(def_map[c][1]) if c in def_map else None)
    return diagnoses


def make_phenotype_label_matrix(phenotypes, stays=None):
    phenotypes = phenotypes[['ICUSTAY_ID', 'HCUP_CCS_2015']].ix[phenotypes.USE_IN_BENCHMARK > 0].drop_duplicates()
    phenotypes['VALUE'] = 1
    phenotypes = phenotypes.pivot(index='ICUSTAY_ID', columns='HCUP_CCS_2015', values='VALUE')
    if stays is not None:
        phenotypes = phenotypes.ix[stays.ICUSTAY_ID.sort_values()]
    return phenotypes.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)


###################################
# Time series preprocessing
###################################

def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):
    var_map = dataframe_from_csv(fn, index_col=None).fillna('').astype(str)
    # var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map.ix[(var_map[variable_column] != '') & (var_map.COUNT > 0)]
    var_map = var_map.ix[(var_map.STATUS == 'ready')]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']].set_index('ITEMID')
    return var_map.rename_axis({variable_column: 'VARIABLE', 'MIMIC LABEL': 'MIMIC_LABEL'}, axis=1)


def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='ITEMID', right_index=True)


def read_variable_ranges(fn, variable_column='LEVEL2'):
    columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    to_rename = dict(zip(columns, [c.replace(' ', '_') for c in columns]))
    to_rename[variable_column] = 'VARIABLE'
    var_ranges = dataframe_from_csv(fn, index_col=None)
    # var_ranges = var_ranges[variable_column].apply(lambda s: s.lower())
    var_ranges = var_ranges[columns]
    var_ranges.rename_axis(to_rename, axis=1, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges.set_index('VARIABLE', inplace=True)
    return var_ranges.ix[var_ranges.notnull().all(axis=1)]


def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:
        return events
    idx = (events.VARIABLE == variable)
    V = events.VALUE[idx]
    V.ix[V < ranges.OUTLIER_LOW[variable]] = np.nan
    V.ix[V > ranges.OUTLIER_HIGH[variable]] = np.nan
    V.ix[V < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
    V.ix[V > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
    events.ix[idx, 'VALUE'] = V
    return events


# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df.VALUE.astype(str)
    idx = v.apply(lambda s: '/' in s)
    v.ix[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.VALUE.astype(str)
    idx = v.apply(lambda s: '/' in s)
    v.ix[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)


# CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.VALUE is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df.VALUE = df.VALUE.astype(str)

    v.ix[(df.VALUE == 'Normal <3 secs') | (df.VALUE == 'Brisk')] = 0
    v.ix[(df.VALUE == 'Abnormal >3 secs') | (df.VALUE == 'Delayed')] = 1
    return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df.VALUE.astype(float)

    ''' The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    ''' The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.VALUE > 1.0)

    ''' The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    '''
    is_str = np.array(map(lambda x: type(x) == str, list(df.VALUE)), dtype=np.bool)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (is_str | (~is_str & (v > 1.0)))

    v.ix[idx] = v[idx] / 100.
    return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.VALUE
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.ix[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.VALUE
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.ix[idx] = np.nan

    v = v.astype(float)
    idx = (v <= 1)
    v.ix[idx] = v[idx] * 100.
    return v


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    v = df.VALUE.astype(float)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'F' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v.ix[idx] = (v[idx] - 32) * 5. / 9
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df.VALUE.astype(float)
    # ounces
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'oz' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'oz' in s.lower())
    v.ix[idx] = v[idx] / 16.
    # pounds
    idx = idx | df.VALUEUOM.fillna('').apply(lambda s: 'lb' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'lb' in s.lower())
    v.ix[idx] = v[idx] * 0.453592
    return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df.VALUE.astype(float)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'in' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'in' in s.lower())
    v.ix[idx] = np.round(v[idx] * 2.54)
    return v


# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous
# Glascow coma scale eye opening
# Glascow coma scale motor response
# Glascow coma scale total
# Glascow coma scale verbal response
# Heart Rate
# Respiratory rate
# Mean blood pressure
clean_fns = {
    'Capillary refill rate': clean_crr,
    'Diastolic blood pressure': clean_dbp,
    'Systolic blood pressure': clean_sbp,
    'Fraction inspired oxygen': clean_fio2,
    'Oxygen saturation': clean_o2sat,
    'Glucose': clean_lab,
    'pH': clean_lab,
    'Temperature': clean_temperature,
    'Weight': clean_weight,
    'Height': clean_height
}


def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = (events.VARIABLE == var_name)
        try:
            events.ix[idx, 'VALUE'] = clean_fn(events.ix[idx])
        except Exception as e:
            print("Exception in clean_events:", clean_fn.__name__, e)
            print("number of rows:", np.sum(idx))
            print("values:", events.ix[idx])
            exit()
    return events.ix[events.VALUE.notnull()]
