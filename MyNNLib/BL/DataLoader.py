import _pickle as cPickle
import gzip
import numpy as np

class DataLoader:
    @staticmethod
    def __vectrized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    @staticmethod
    def load() :
        f = gzip.open('C:\Projects\ML\MLRepo\data\mnist.pkl.gz', 'rb')
        tr_d, va_d, te_d = cPickle.load(f, encoding='latin1')
        f.close()
        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_results = [DataLoader.__vectrized_result(y) for y in tr_d[1]]
        training_data = np.array(list(zip(training_inputs, training_results)))
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = np.array(list(zip(validation_inputs, va_d[1])))
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = np.array(list(zip(test_inputs, te_d[1])))
        return [training_data, validation_data, test_data]
