import os
import numpy as np
import random


class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if (seed is not None):
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        if not hasattr(self, '_current_index'):
            self._current_index = 0
        to_read_index = self._current_index
        self._current_index += 1
        if (self._current_index == self.get_number_of_examples()):
            self._current_index = 0
        return self.read_example(to_read_index)


class DecompensationReader(Reader):
    r"""
    Reader for decompensation prediction task.
    
    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listilfe : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """

    def __init__(self, dataset_dir, listfile=None):
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), int(y)) for (x, t, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        r"""
        Reads the example with given index.

        Parameters
        ----------
        index : int
            Index of the line of the listfile to read (counting starts from 0).
        
        Returns (X, t, y, header)
        -------
        X : np.array
            2D array containing all events. Each row corresponds to a moment.
            First coloumn is the time and other columns correspond to different
            variables.
        t : float
            Lenght of the data in hours. Note, in general, it is not eqaul to the
            timestamp of last event.
        y : int (0 or 1)
            Mortality within next 24 hours.
        header : array of strings
            Names of the columns. The ordering of the columns is always the same.
        """
        if (index < 0 or index >= len(self._data)):
            raise ValueError("Index must be from 0 (inclusive) to number of examples (exclusive).")

        t = self._data[index][1]
        (X, header) = self._read_timeseries(self._data[index][0], t)
        y = self._data[index][2]

        return (X, t, y, header)


class InHospitalMortalityReader(Reader):
    r"""
    Reader for in-hospital moratality prediction task.
    
    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listilfe : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    period_length : float
        Length of the period (in hours) from which the prediction is done.
    """

    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        r"""
        Reads the example with given index.

        Parameters
        ----------
        index : int
            Index of the line of the listfile to read (counting starts from 0).

        Returns (X, t, y, header)
        -------
        X : np.array
            2D array containing all events. Each row corresponds to a moment.
            First coloumn is the time and other columns correspond to different
            variables.
        t : float
            Lenght of the data in hours. Note, in general, it is not eqaul to the
            timestamp of last event.
        y : int (0 or 1)
            In-hospital mortality.
        header : array of strings
            Names of the columns. The ordering of the columns is always the same.
        """
        if (index < 0 or index >= len(self._data)):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        (X, header) = self._read_timeseries(self._data[index][0])
        y = self._data[index][1]

        return (X, self._period_length, y, header)


class LengthOfStayReader(Reader):
    r"""
    Reader for length of stay prediction task.

    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listilfe : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """

    def __init__(self, dataset_dir, listfile=None):
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), float(y)) for (x, t, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        r"""
        Reads the example with given index.

        Parameters
        ----------
        index : int
            Index of the line of the listfile to read (counting starts from 0).

        Returns (X, t, y, header)
        -------
        X : np.array
            2D array containing all events. Each row corresponds to a moment.
            First coloumn is the time and other columns correspond to different
            variables.
        t : float
            Lenght of the data in hours. Note, in general, it is not eqaul to the
            timestamp of last event.
        y : float
            Remaining time in ICU.
        header : array of strings
            Names of the columns. The ordering of the columns is always the same.
        """
        if (index < 0 or index >= len(self._data)):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
    
        t = self._data[index][1]
        (X, header) = self._read_timeseries(self._data[index][0], t)
        y = self._data[index][2]

        return (X, t, y, header)


class PhenotypingReader(Reader):
    r"""
    Reader for phenotype classification task.

    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listilfe : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """

    def __init__(self, dataset_dir, listfile=None):
        Reader.__init__(self, dataset_dir, listfile)
        self._listfile_header = self._data[0]
        self._data = self._data[1:]
        self._data = [line.split(',') for line in self._data]
        self._data = [(mas[0], float(mas[1]), map(int, mas[2:])) for mas in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        r"""
        Reads the example with given index.

        Parameters
        ----------
        index : int
            Index of the line of the listfile to read (counting starts from 0).

        Returns (X, t, y, header)
        -------
        X : np.array
            2D array containing all events. Each row corresponds to a moment.
            First coloumn is the time and other columns correspond to different
            variables.
        t : float
            Lenght of the data in hours. Note, in general, it is not eqaul to the
            timestamp of last event.
        y : array of ints
            Phenotype labels.
        header : array of strings
            Names of the columns. The ordering of the columns is always the same.
        """
        if (index < 0 or index >= len(self._data)):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        (X, header) = self._read_timeseries(self._data[index][0])
        y = self._data[index][2]

        return (X, self._data[index][1], y, header)


class MultitaskReader(Reader):
    r"""
    Reader for multitask.

    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listilfe : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """
    
    def __init__(self, dataset_dir, listfile=None):
        Reader.__init__(self, dataset_dir, listfile)
        self._listfile_header = self._data[0]
        self._data = self._data[1:]
        self._data = [line.split(',') for line in self._data]
        
        def process_ihm(ihm):
            return map(int, ihm.split(';'))
        
        def process_los(los):
            los = los.split(';')
            return (map(int, los[:len(los)/2]), map(float, los[len(los)/2:]))
        
        def process_ph(ph):
            return map(int, ph.split(';'))
        
        def process_decomp(decomp):
            decomp = decomp.split(';')
            return (map(int, decomp[:len(decomp)/2]), map(int, decomp[len(decomp)/2:]))
        
        self._data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(ph), process_decomp(decomp))
                            for fname, t, ihm, los, ph, decomp in self._data]
    
    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    
    def read_example(self, index):
        r"""
        Reads the example with given index.

        Parameters
        ----------
        index : int
            Index of the line of the listfile to read (counting starts from 0).

        Returns (X, t, ihm, los, ph, decomp, header)
        -------
        X : np.array
            2D array containing all events. Each row corresponds to a moment.
            First coloumn is the time and other columns correspond to different
            variables.
        t : float
            Lenght of the data in hours. Note, in general, it is not eqaul to the
            timestamp of last event.
        ihm : array
            Array of 3 integers: [pos, mask, label].
        los : array
            Array of 2 arrays: [masks, labels].
        ph : array
            Array of 25 binary integers (phenotype labels).
        decomp : array
            Array of 2 arrays: [masks, labels].
        header : array of strings
            Names of the columns. The ordering of the columns is always the same.
        """
        if (index < 0 or index >= len(self._data)):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
        (X, header) = self._read_timeseries(self._data[index][0])
        return [X] + list(self._data[index][1:]) + [header]
