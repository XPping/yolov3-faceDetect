# coding: utf-8

import os
import numpy as np

def set_gpu(gpu_id):
    print("Using GPU id = {}".format(gpu_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)

# make dirs for model output
def make_dirs(names):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)

class ReduceLR(object):
    def __init__(self, factor=0.1, patience=10, min_lr=0):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.wait = 0
        self.best_val_loss = np.Inf

    def __call__(self, current_val_loss, current_lr):
        if self.best_val_loss > current_val_loss:
            self.wait += 1
        new_lr = current_lr
        if self.wait >= self.patience:
            new_lr = current_lr * self.factor
            new_lr = max(new_lr, self.min_lr)
            self.wait = 0
        return new_lr