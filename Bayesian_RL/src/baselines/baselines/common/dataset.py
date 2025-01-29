import numpy as np


class Dataset(object):

    def __init__(
            self,
            data_map,
            deterministic=False,
            shuffle=True):
        
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]
        
        self._next_id = 0

    