import os
import numpy as np
import random
import cPickle as pickle


class Discretizer():
    
    def __init__(self, timestep=0.8, store_masks=True, imput_strategy='zero', start_time='zero'):
        
        self._id_to_channel = [
            'Capillary refill rate',
            'Diastolic blood pressure',
            'Fraction inspired oxygen',
            'Glascow coma scale eye opening',
            'Glascow coma scale motor response',
            'Glascow coma scale total',
            'Glascow coma scale verbal response',
            'Glucose',
            'Heart Rate',
            'Height',
            'Mean blood pressure',
            'Oxygen saturation',
            'Respiratory rate',
            'Systolic blood pressure',
            'Temperature',
            'Weight',
            'pH']

        self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
       
        self._is_categorical_channel = {
            'Capillary refill rate': True,
            'Diastolic blood pressure': False,
            'Fraction inspired oxygen': False,
            'Glascow coma scale eye opening': True,
            'Glascow coma scale motor response': True,
            'Glascow coma scale total': True,
            'Glascow coma scale verbal response': True,
            'Glucose': False,
            'Heart Rate': False,
            'Height': False,
            'Mean blood pressure': False,
            'Oxygen saturation': False,
            'Respiratory rate': False,
            'Systolic blood pressure': False,
            'Temperature': False,
            'Weight': False,
            'pH': False}

        self._possible_values = {
            'Capillary refill rate': ['0.0', '1.0'],
            'Diastolic blood pressure': [],
            'Fraction inspired oxygen': [],
            'Glascow coma scale eye opening': ['To Pain',
                '3 To speech',
                '1 No Response',
                '4 Spontaneously',
                'None',
                'To Speech',
                'Spontaneously',
                '2 To pain'],
            'Glascow coma scale motor response': ['1 No Response',
                '3 Abnorm flexion',
                'Abnormal extension',
                'No response',
                '4 Flex-withdraws',
                'Localizes Pain',
                'Flex-withdraws',
                'Obeys Commands',
                'Abnormal Flexion',
                '6 Obeys Commands',
                '5 Localizes Pain',
                '2 Abnorm extensn'],
            'Glascow coma scale total': ['11',
                '10',
                '13',
                '12',
                '15',
                '14',
                '3',
                '5',
                '4',
                '7',
                '6',
                '9',
                '8'],
            'Glascow coma scale verbal response': ['1 No Response',
                'No Response',
                'Confused',
                'Inappropriate Words',
                'Oriented',
                'No Response-ETT',
                '5 Oriented',
                'Incomprehensible sounds',
                '1.0 ET/Trach',
                '4 Confused',
                '2 Incomp sounds',
                '3 Inapprop words'],
            'Glucose': [],
            'Heart Rate': [],
            'Height': [],
            'Mean blood pressure': [],
            'Oxygen saturation': [],
            'Respiratory rate': [],
            'Systolic blood pressure': [],
            'Temperature': [],
            'Weight': [],
            'pH': []
        }
        
        self._normal_values = {
            'Capillary refill rate': '0.0',
            'Diastolic blood pressure': '59.0',
            'Fraction inspired oxygen': '0.21',
            'Glascow coma scale eye opening': '4 Spontaneously',
            'Glascow coma scale motor response': '6 Obeys Commands',
            'Glascow coma scale total': '15',
            'Glascow coma scale verbal response': '5 Oriented',
            'Glucose': '128.0',
            'Heart Rate': '86',
            'Height': '170.0',
            'Mean blood pressure': '77.0',
            'Oxygen saturation': '98.0',
            'Respiratory rate': '19',
            'Systolic blood pressure': '118.0',
            'Temperature': '36.6',
            'Weight': '81.0',
            'pH': '7.4',
        }

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._imput_strategy = imput_strategy
        
        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0
        
    def transform(self, X, header=None, end=None):
        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6
    
        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps
        
        if (self._start_time == 'relative'):
            first_time = ts[0]
        elif (self._start_time == 'zero'):
            first_time = 0
        else:
            raise ValueError("start_time is invalid")
        
        if (end == None):
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time
        
        N_bins = int(max_hours / self._timestep + 1.0 - eps)
        
        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if (self._is_categorical_channel[channel]):
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]
        
        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0
        
        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if (self._is_categorical_channel[channel]):
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)
        
        for row in X:
            t = float(row[0]) - first_time
            if (t > max_hours + eps):
                continue
            bin_id = int(t / self._timestep - eps)
            assert(bin_id >= 0 and bin_id < N_bins)
            
            for j in range(1, len(row)):
                if (row[j] == ""):
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if (mask[bin_id][channel_id] == 1):
                    unused_data += 1
                mask[bin_id][channel_id] = 1
                
                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]
        
        # impute missing values
        
        if (self._imput_strategy not in ['zero', 'normal_value', 'previous', 'next']):
            raise ValueError("impute strategy is invalid")
        
        if (self._imput_strategy in ['normal_value', 'previous']):
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if (mask[bin_id][channel_id] == 1):
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if (self._imput_strategy == 'normal_value'):
                        imputed_value = self._normal_values[channel]
                    if (self._imput_strategy == 'previous'):
                        if (len(prev_values[channel_id]) == 0):
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if (self._imput_strategy == 'next'):
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if (mask[bin_id][channel_id] == 1):
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if (len(prev_values[channel_id]) == 0):
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)
        
        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)
    
        if (self._store_masks):
            data = np.hstack([data, mask.astype(np.float32)])
        
        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if (self._is_categorical_channel[channel]):
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)
        
        if (self._store_masks):
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)
        
        new_header = ",".join(new_header)
        
        return (data, new_header)

    def print_statistics(self):
        print "statistics of discretizer:"
        print "\tconverted %d examples" % self._done_count
        print "\taverage unused data = %.2f percent" % (100.0 * self._unused_data_sum / self._done_count)
        print "\taverage empty  bins = %.2f percent" % (100.0 * self._empty_bins_sum / self._done_count)


class Normalizer():
    
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if (fields is not None):
            self._fields = [col for col in fields]
        
        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0
    
    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if (self._sum_x is None):
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x**2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x**2, axis=0)
    
    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(1.0 / (N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=-1)
    
    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            dct = pickle.load(load_file)
            self._means = dct['means']
            self._stds = dct['stds']
    
    def transform(self, X):
        if (self._fields is None):
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret
