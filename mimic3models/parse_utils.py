from __future__ import print_function
import re


def parse_task(log):
    if re.search('ihm_C', log):
        return 'multitask'
    if re.search('partition', log):
        return 'los'
    if re.search('deep_supervision', log):
        return 'decomp'
    if re.search('ave_auc_micro', log):
        return 'pheno'
    if re.search('AUC of ROC', log):
        return'ihm'
    return None


def get_loss(log, lossname):
    """ Options for lossname: 'loss', 'ihm_loss', 'decomp_loss', 'pheno_loss', 'los_loss'
    """
    train = re.findall('[^_]{}: ([0-9.]+)'.format(lossname), log)
    train = map(float, train)
    val = re.findall('val_{}: ([0-9.]+)'.format(lossname), log)
    val = map(float, val)
    if len(train) > len(val):
        assert len(train) - 1 == len(val)
        train = train[:-1]
    return train, val


def parse_metrics(metric, log):
    ret = re.findall('{} = (.*)\n'.format(metric), log)
    ret = map(float, ret)
    if len(ret) % 2 == 1:
        ret = ret[:-1]
    return ret[::2], ret[1::2]


def parse_last_state(log):
    ret = re.findall('saving model to (.*.state)\n', log)
    if len(ret) == 0:
        return None
    return ret[-1]


def parse_network(log):
    ret = re.search("network='([^']*)'", log)
    return ret.group(1)


def parse_prefix(log):
    ret = re.search("prefix='([^']*)'", log)
    return ret.group(1)


def parse_dim(log):
    ret = re.search("dim=([0-9]*)", log)
    return int(ret.group(1))


def parse_size_coef(log):
    ret = re.search('size_coef=([\.0-9]*)', log)
    return ret.group(1)


def parse_depth(log):
    ret = re.search('depth=([0-9]*)', log)
    return int(ret.group(1))


def parse_ihm_C(log):
    ret = re.search('ihm_C=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None


def parse_decomp_C(log):
    ret = re.search('decomp_C=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None


def parse_los_C(log):
    ret = re.search('los_C=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None


def parse_pheno_C(log):
    ret = re.search('pheno_C=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None


def parse_dropout(log):
    ret = re.search('dropout=([\.0-9]*)', log)
    return float(ret.group(1))


def parse_partition(log):
    ret = re.search("partition='([^']*)'", log)
    if ret:
        return ret.group(1)
    return None


def parse_deep_supervision(log):
    ret = re.search('deep_supervision=(.*),', log)
    if ret:
        return ret.group(1) == 'True'
    return False


def parse_target_repl_coef(log):
    ret = re.search('target_repl_coef=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None


def parse_epoch(state):
    ret = re.search('.*(chunk|epoch)([0-9]*).*', state)
    return int(ret.group(2))
