import random
import torch
import torch.multiprocessing as multiprocessing
from torch._C import _set_worker_signal_handlers, _update_worker_pids, \
    _remove_worker_pids, _error_if_any_worker_fails
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
import signal
import functools
import collections
import re
import sys
import threading
import traceback
import os
import time
from torch._six import string_classes, int_classes, FileNotFoundError

class imgVoxDataloader(torch.utils.data.DataLoader):
    
    __initialized = False

    def __init__(self, dataset,dataset1, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False, collate_fn=torch.utils.data.dataloader.default_collate, timeout=0, worker_init_fn=None):
        self.dataset = dataset
        self.dataset1 = dataset1
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                    sampler1 = RandomSampler(dataset1)
                else:
                    sampler = SequentialSampler(dataset)
                    sampler1 = SequentialSampler(dataset1)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
            batch_sampler1 = BatchSampler(sampler1, batch_size, drop_last)
            
        self.sampler = sampler #(sampler, sampler1)
        self.sampler1 = sampler1
        self.batch_sampler = batch_sampler #(batch_sampler,batch_sampler1)
        self.batch_sampler1 = batch_sampler1
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(imgVoxDataloader, self).__setattr__(attr, val)

    def __iter__(self):
        return torch.utils.data.dataloader._DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler[0])