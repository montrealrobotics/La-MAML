### We directly copied the metrics.py model file from the GEM project https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import ipdb
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import torch


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def confusion_matrix(result_t, result_a, log_dir, fname=None):
    nt, changes = task_changes(result_t)
    fname = os.path.join(log_dir, fname)

    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    result = result_a[(torch.LongTensor(changes))]

    # acc[t] equals result[t,t]
    acc = result.diag()
    fin = result[nt - 1]
    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    if fname is not None:
        f = open(fname, 'w')

        print(' '.join(['%.4f' % r for r in baseline]), file=f)
        print('|', file=f)
        for row in range(result.size(0)):
            print(' '.join(['%.4f' % r for r in result[row]]), file=f)
        print('', file=f)
        print('Diagonal Accuracy: %.4f' % acc.mean(), file=f)
        print('Final Accuracy: %.4f' % fin.mean(), file=f)
        print('Backward: %.4f' % bwt.mean(), file=f)
        print('Forward:  %.4f' % fwt.mean(), file=f)
        f.close()

    colors = cm.nipy_spectral(np.linspace(0, 1, len(result)))
    figure = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    data = np.array(result_a)
    for i in range(len(data[0])):
        plt.plot(range(data.shape[0]), data[:,i], label=str(i), color=colors[i], linewidth=2)
        
    plt.savefig(log_dir + '/' + 'task_wise_accuracy.png')

    stats = []
    stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())

    return stats
