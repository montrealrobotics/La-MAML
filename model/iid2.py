import torch

import numpy as np
import random

import model.meta.learner as Learner
import model.meta.modelfactory as mf
import ipdb
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("once")

"""
Multi task
    big batch size, set increment 100 so that it is treated as 1 task with all classes in the dataset
    inference time for acc eval, use offsets
"""

class Net(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.args = args
        self.nt = n_tasks

        self.n_feat = n_outputs
        self.n_classes = n_outputs

        arch = args.arch
        nl, nh = args.n_layers, args.n_hiddens
        config = mf.ModelFactory.get_model(model_type = arch, sizes = [n_inputs] + [nh] * nl + [n_outputs],
                                                dataset = args.dataset, args=args)
        self.net = Learner.Learner(config, args)

        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # setup losses
        self.loss = torch.nn.CrossEntropyLoss()

        self.gpu = args.cuda
        self.nc_per_task = int(n_outputs / n_tasks)
        self.n_outputs = n_outputs

    def compute_offsets(self, task):
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)

    def take_multitask_loss(self, bt, logits, y):
        loss = 0.0
        for i, ti in enumerate(bt):
            offset1, offset2 = self.compute_offsets(ti)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)

    def forward(self, x, t):                                  
                                                
        output = self.net.forward(x)

        # make sure we predict classes within the current task
        if torch.unique(t).shape[0] == 1:
            offset1, offset2 = self.compute_offsets(t[0].item())
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        else:
            for i in range(len(t)):
                offset1, offset2 = self.compute_offsets(t[i])
                if offset1 > 0:
                    output[i, :offset1].data.fill_(-10e10)
                if offset2 < self.n_outputs:
                    output[i, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, y, t):
        self.net.train()

        self.net.zero_grad()
        logits = self.net.forward(x)
        loss = self.take_multitask_loss(t, logits, y) 
        loss.backward()
        self.opt.step()

        return loss.item()