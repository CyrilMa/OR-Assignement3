# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:51:39 2018

@author: Cyril
"""

import numpy as np

# Developer: Alejandro Debus
# Email: aledebus@gmail.com

def partitions(number, k):
    '''
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

def get_indices(n_splits = 3, subjects = 145):
    '''
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    l = partitions(subjects, n_splits)
    fold_sizes = l
    indices = np.arange(subjects).astype(int)
    np.random.shuffle(indices)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def k_folds(n_splits = 3, subjects = 1185):
    '''
    Generates folds for cross validation
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    
    indices = np.arange(subjects).astype(int)
    np.random.shuffle(indices)
    for test_idx in get_indices(n_splits, subjects):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx
