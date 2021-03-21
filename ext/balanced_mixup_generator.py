import numpy as np
from .mixup_generator import MixupGenerator

def _flatten_y_if_onehot(y):
    """De-one-hot y, i.e. [0,1,0,0,...] to 1 for all y."""
    return y if len(np.array(y).shape) == 1 else [np.argmax(one) for one in y]

def _randam_shuffled(np_array):
    np.random.shuffle(np_array)
    return np_array

def _multiple_of_n(a, n_boundary):
    """Calculates ceil like operation to make 'a' to be multiple of n boundary."""
    ceil_like = (a + n_boundary - 1) // n_boundary
    return ceil_like * n_boundary

def _interleaved_stack(arrays):
    # https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    # a = np.array([1,3,5])
    # b = np.array([2,4,6])
    # c = np.array([1,3,5]) + 10
    # d = np.array([2,4,6]) + 20
    # np.vstack((a,b,c,d)).reshape((-1,),order='F')
    # --> array([ 1,  2, 11, 22,  3,  4, 13, 24,  5,  6, 15, 26])
    return np.vstack(arrays).reshape((-1,), order='F')

class BalancedMixupGenerator(MixupGenerator):
    """A Generator that always generate class balanced, mixup-applied batch."""

    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None,
                 verbose=0):
        super().__init__(X_train, y_train, batch_size=batch_size, 
                         alpha=alpha, shuffle=shuffle, datagen=datagen)

        self.class_train = np.array(_flatten_y_if_onehot(y_train))
        self.classes = sorted(list(set(self.class_train)))
        self.class_rnd_idx = {cur_cls: _randam_shuffled(np.where(self.class_train == cur_cls)[0])
                              for cur_cls in self.classes}
        self.verbose = verbose

        assert batch_size >= len(self.classes), \
            'Use normal MixupGenerator if you wish small batch size than # of classes.' \
            + 'batch_size={} num_classes={}'.format(batch_size, len(self.classes))
        assert self.shuffle, 'Unfortunately non-shuffle is not supported...'

    def _ensure_unique_class_start(self, indexes):
        """Ensures that adjoining batches don't start from the same class.
           Avoiding mixing up the same class samples."""

        for i in range(0, len(indexes) - self.batch_size, self.batch_size):
            j = i + self.batch_size
            if self.class_train[indexes[i]] == self.class_train[indexes[j]]:
                indexes[j:j+self.batch_size] = np.roll(indexes[j:j+self.batch_size], 1)
        return indexes

    def get_exploration_order(self):
        # Calculate # of samples per class which can accommodate all the samples.
        max_size_of_class_sample = np.max([len(v) for v in self.class_rnd_idx.values()])
        n_samples_per_class = np.max([self.sample_num // len(self.classes),
                                      max_size_of_class_sample])
        # It has to be multiple of 2 * batch_size.
        n_samples_per_class = _multiple_of_n(n_samples_per_class, 2 * self.batch_size)
        # Make even number of tiled samples for all classes.
        cls_idx = self.class_rnd_idx
        class_indexes = {cls: np.tile(cls_idx[cls],
                                      int(np.ceil(n_samples_per_class / len(cls_idx[cls])))
                                      )[:n_samples_per_class] # tiled samples per class
                         for cls in self.classes}
        # Merge interleaving to make a single index list consist of multiple of 2 * batch_size.
        indexes = _interleaved_stack(list(class_indexes.values()))
        # Ensure adjoining batch start from different class
        # so that mixing up will not mix the same class samples.
        #print(np.reshape(self.class_train[indexes], (len(indexes)//12, 12)))
        indexes = self._ensure_unique_class_start(indexes)
        #print(np.reshape(self.class_train[indexes], (len(indexes)//12, 12)))

        return indexes
