from __future__ import absolute_import
from __future__ import print_function

import argparse
from mimic3models import parse_utils
import json
import numpy as np


def check_decreasing(a, k, eps):
    if k >= len(a):
        return False
    pos = len(a) - 1
    for i in range(k):
        if a[pos] > a[pos - 1] + eps:
            return False
        pos -= 1
    return True


def process_single(filename, verbose, select):
    if verbose:
        print("Processing log file: {}".format(filename))

    with open(filename, 'r') as fin:
        log = fin.read()
    task = parse_utils.parse_task(log)

    if task is None:
        print("Task is not detected: {}".format(filename))
        return None

    if verbose:
        print("\ttask = {}".format(task))

    if task == 'multitask' or task == 'pheno':
        metric = 'ave_auc_macro'
    elif task == 'ihm' or task == 'decomp':
        metric = 'AUC of ROC'
    elif task == 'los':
        metric = 'Cohen kappa score'
    else:
        assert False

    train_metrics, val_metrics = parse_utils.parse_metrics(log, metric)
    if len(train_metrics) == 0:
        print("Less than one epoch: {}".format(filename))
        return None
    last_train = train_metrics[-1]
    last_val = val_metrics[-1]

    if verbose:
        print("\tlast train = {}, last val = {}".format(last_train, last_val))

    rerun = True
    if task == 'ihm':
        if last_val < 0.83 and last_train > 0.88:
            rerun = False
        if last_val < 0.84 and last_train > 0.89:
            rerun = False
        if last_val < 0.85 and last_train > 0.9:
            rerun = False
    elif task == 'decomp':
        if last_val < 0.85 and last_train > 0.89:
            rerun = False
        if last_val < 0.87 and last_train > 0.9:
            rerun = False
        if last_val < 0.88 and last_train > 0.92:
            rerun = False
    elif task == 'pheno' or task == 'multitask':
        if last_val < 0.75 and last_train > 0.77:
            rerun = False
        if last_val < 0.76 and last_train > 0.79:
            rerun = False
    elif task == 'los':
        if last_val < 0.35 and last_train > 0.42:
            rerun = False
        if last_val < 0.38 and last_train > 0.44:
            rerun = False
    else:
        assert False

    # check if val_metrics is decreasing
    if task in ['ihm', 'decomp', 'pheno', 'multitask']:
        n_decreases = 3
    else:  # 'los'
        n_decreases = 5

    if check_decreasing(val_metrics, n_decreases, 0.001):
        rerun = False

    # check if maximum value for validation was very early
    if task in ['ihm', 'decomp', 'pheno', 'multitask']:
        tol = 0.01
    else:  # 'los'
        tol = 0.03
    val_max = max(val_metrics)
    val_max_pos = np.argmax(val_metrics)
    if len(val_metrics) - val_max_pos >= 8 and val_max - last_val > tol:
        rerun = False

    if not select:
        rerun = True

    if verbose:
        print("\trerun = {}".format(rerun))

    if not rerun:
        return None

    # need to rerun
    last_state = parse_utils.parse_last_state(log)
    if last_state is None:
        print("Last state is not parsed: {}".format(filename))
        return None

    n_epochs = parse_utils.parse_epoch(last_state)

    if verbose:
        print("\tlast state = {}".format(last_state))

    network = parse_utils.parse_network(log)

    prefix = parse_utils.parse_prefix(log)
    if prefix == '':
        prefix = 'r2'
    elif not str.isdigit(prefix[-1]):
        prefix += '2'
    else:
        prefix = prefix[:-1] + str(int(prefix[-1]) + 1)

    dim = parse_utils.parse_dim(log)
    size_coef = parse_utils.parse_size_coef(log)
    depth = parse_utils.parse_depth(log)

    ihm_C = parse_utils.parse_ihm_C(log)
    decomp_C = parse_utils.parse_decomp_C(log)
    los_C = parse_utils.parse_los_C(log)
    pheno_C = parse_utils.parse_pheno_C(log)

    dropout = parse_utils.parse_dropout(log)
    partition = parse_utils.parse_partition(log)
    deep_supervision = parse_utils.parse_deep_supervision(log)
    target_repl_coef = parse_utils.parse_target_repl_coef(log)

    batch_size = parse_utils.parse_batch_size(log)

    command = "python -u main.py --network {} --prefix {} --dim {}"\
              " --depth {} --epochs 100 --batch_size {} --timestep 1.0"\
              " --load_state {}".format(network, prefix, dim, depth,  batch_size, last_state)

    if network.find('channel') != -1:
        command += ' --size_coef {}'.format(size_coef)

    if ihm_C:
        command += ' --ihm_C {}'.format(ihm_C)

    if decomp_C:
        command += ' --decomp_C {}'.format(decomp_C)

    if los_C:
        command += ' --los_C {}'.format(los_C)

    if pheno_C:
        command += ' --pheno_C {}'.format(pheno_C)

    if dropout > 0.0:
        command += ' --dropout {}'.format(dropout)

    if partition:
        command += ' --partition {}'.format(partition)

    if deep_supervision:
        command += ' --deep_supervision'

    if (target_repl_coef is not None) and target_repl_coef > 0.0:
        command += ' --target_repl_coef {}'.format(target_repl_coef)

    return {"command": command,
            "train_max": np.max(train_metrics),
            "train_max_pos": np.argmax(train_metrics),
            "val_max": np.max(val_metrics),
            "val_max_pos": np.argmax(val_metrics),
            "last_train": last_train,
            "last_val": last_val,
            "n_epochs": n_epochs,
            "filename": filename}


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('logs', type=str, nargs='+')
    argparser.add_argument('--verbose', type=int, default=0)
    argparser.add_argument('--select', dest='select', action='store_true')
    argparser.add_argument('--no-select', dest='select', action='store_false')
    argparser.set_defaults(select=True)
    args = argparser.parse_args()

    if not isinstance(args.logs, list):
        args.logs = [args.logs]

    rerun = []
    for log in args.logs:
        if log.find(".log") == -1:  # not a log file or is a not renamed log file
            continue
        ret = process_single(log, args.verbose, args.select)
        if ret:
            rerun += [ret]
    rerun = sorted(rerun, key=lambda x: x["last_val"], reverse=True)

    print("Need to rerun {} / {} models".format(len(rerun), len(args.logs)))
    print("Saving the results in rerun_output.json")
    with open("rerun_output.json", 'w') as fout:
        json.dump(rerun, fout)

    print("Saving commands in rerun_commands.sh")
    with open("rerun.sh", 'w') as fout:
        for a in rerun:
            fout.write(a['command'] + '\n')

    print("Saving filenames in rerun_filenames.txt")
    with open("rerun_filenames.txt", 'w') as fout:
        for a in rerun:
            fout.write(a['filename'] + '\n')


if __name__ == '__main__':
    main()
