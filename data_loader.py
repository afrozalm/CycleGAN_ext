import numpy as np


class IndexIterator(object):
    def __init__(self, limit, batch_size):
        self.limit = limit
        self.batch_size = batch_size
        self.idx = 0
        self.update_perm()

    def update_perm(self):
        self.curr_perm = np.random.permutation(self.limit)

    def __iter__(self):
        return self

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.idx + batch_size < self.limit:
            output = self.curr_perm[self.idx: self.idx + batch_size]
            self.idx += batch_size
        else:
            self.update_perm()
            output = self.curr_perm[: batch_size]
            self.idx = batch_size
        return output


class GroupIterator(object):
    def __init__(self, limit, members, data_ptrs, batch_size):
        self.limit = limit
        self.data_ptrs = data_ptrs
        self.members = members
        self.batch_size = batch_size
        self.idx_iterator = IndexIterator(limit=limit,
                                          batch_size=batch_size)

    def add_member(self, member, data_ptr):
        assert isinstance(data_ptr, np.ndarray)
        assert data_ptr.shape[0] == self.limit
        self.members.append(member)
        self.data_ptrs.append(data_ptr)

    def next(self, batch_size=None):
        '''
        This function returns a list of ndarrays
        in the respective order of data pointers
        added to this object.

        See self.members to see the output order
        '''
        output = []
        curr_perm = self.idx_iterator.next(batch_size)
        for data_ptr in self.data_ptrs:
            output.append(data_ptr[curr_perm])

        return output


class DataIterator(object):
    def __init__(self, data_ptr, batch_size):
        if not isinstance(data_ptr, np.ndarray):
            raise TypeError('np.ndarray expected')
        else:
            self.data_ptr = data_ptr
            self.limit = data_ptr.shape[0]
            self.batch_size = batch_size
            self.idx_iterator = IndexIterator(limit=self.limit,
                                              batch_size=self.batch_size)

    def __iter__(self):
        return self

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        return self.data_ptr[self.idx_iterator.next(batch_size)]

    def get_idx_samples(self, perm):
        assert isinstance(perm, np.ndarray)
        return self.data[perm]


class DataLoader(object):
    def __init__(self, batch_size):
        '''this is init'''
        self.datasets = {}
        self.group_pool = {}
        self.batch_size = batch_size

    def add_dataset(self, name, data_ptr, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if not isinstance(name, str):
            raise TypeError('name should be string type')
        elif name in self.datasets:
            raise NameError(name +
                            ' already exists as a dataset. Give new name')
        else:
            self.datasets[name] = DataIterator(data_ptr=data_ptr,
                                               batch_size=batch_size)

    def next_batch(self, name, batch_size=None):
        return self.datasets[name].next(batch_size)

    def link_datasets(self, group_name, members):
        assert group_name not in self.group_pool
        limit = self.datasets[members[0]].limit
        data_ptrs = []

        for member in members:
            assert member in self.datasets
            assert limit == self.datasets[member].limit
            data_ptrs.append(self.datasets[member].data_ptr)

        self.group_pool[group_name] = GroupIterator(limit=limit,
                                                    members=members,
                                                    data_ptrs=data_ptrs,
                                                    batch_size=self.batch_size)

    def next_group_batch(self, group_name, batch_size=None):
        return self.group_pool[group_name].next(batch_size)
