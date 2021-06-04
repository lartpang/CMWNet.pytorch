# -*- coding: utf-8 -*-
# @Time    : 2021/6/3
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang


class TrainIterator:
    def __init__(self, start_epoch=0, *, num_iters, data_iterable):
        """
        For uniting the epoch_loop and the batch_loop into a single loop

        :param start_epoch:
        :param num_iters:
        :param data_iterable:
        """
        self._epoch_length = len(data_iterable)
        assert self._epoch_length != 0

        self._num_iters = num_iters
        self._data_iterable = data_iterable
        self._data_iterator = None
        self._curr_iter = start_epoch * self._epoch_length
        self._is_first_iter_in_epoch = False
        self._is_last_iter_in_epoch = False
        self._is_last_iter = False
        self._curr_epoch = -1

    def __repr__(self):
        formatted_string = [
            f"{self.__class__.__name__}: (\n",
            f"epoch_length: {self._epoch_length}\n",
            f"num_iters: {self._num_iters}\n",
            f"curr_iter: {self._curr_iter}\n",
            f"curr_epoch: {self._curr_epoch}\n)",
        ]
        return "\t".join(formatted_string)

    @property
    def is_first_iter_in_epoch(self):
        return self._is_first_iter_in_epoch

    @property
    def is_last_iter_in_epoch(self):
        return self._is_last_iter_in_epoch

    @property
    def is_last_iter(self):
        return self._is_last_iter

    @property
    def curr_epoch(self):
        return self._curr_epoch

    def __iter__(self):
        """
        :return: a iterator object
        """
        return self

    def in_first_iter_of_epoch(self):
        # one epoch will start
        self._is_first_iter_in_epoch = True
        self._curr_epoch += 1
        self._data_iterator = iter(self._data_iterable)

    def in_last_iter_of_epoch(self):
        self._is_last_iter_in_epoch = True

    def in_last_iter(self):
        self._is_last_iter = True

    def __next__(self):
        curr_iter = self._curr_iter
        if curr_iter >= self._num_iters:
            raise StopIteration

        self._is_last_iter_in_epoch = False
        self._is_first_iter_in_epoch = False
        if curr_iter % self._epoch_length == 0:
            self.in_first_iter_of_epoch()
        if (curr_iter + 1) % self._epoch_length == 0:
            self.in_last_iter_of_epoch()
        if (curr_iter + 1) == self._num_iters:
            self.in_last_iter()

        data = next(self._data_iterator)

        iter_in_epoch = curr_iter - self._curr_epoch * self._epoch_length

        self._curr_iter += 1
        return curr_iter, iter_in_epoch, data
