# An implementation of MER Algorithm 1 from https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import sys
import ipdb
from copy import deepcopy
import warnings
import model.meta.learner as Learner
import model.meta.modelfactory as mf
warnings.filterwarnings("ignore")

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.args = args
        nl, nh = args.n_layers, args.n_hiddens

        self.is_cifar = (args.dataset == 'cifar100' or args.dataset == 'tinyimagenet')
        config = mf.ModelFactory.get_model(args.arch, sizes=[n_inputs] + [nh] * nl + [n_outputs], dataset=args.dataset, args=args)
        self.net = Learner.Learner(config, args=args)

        self.netforward = self.net.forward

        self.bce = torch.nn.CrossEntropyLoss()

        self.n_outputs = n_outputs
        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)
        self.batchSize = int(args.replay_batch_size)

        self.memories = args.memories
        self.steps = int(args.batches_per_example)
        self.beta = args.beta
        self.gamma = args.gamma

        # allocate buffer
        self.M = []
        self.age = 0

        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()


    def forward(self, x, t):
        output = self.netforward(x)
        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def getBatch(self,x,y,t):
        if(x is not None):
            xi = Variable(torch.from_numpy(np.array(x))).float().unsqueeze(0) #.view(1,-1)
            yi = Variable(torch.from_numpy(np.array(y))).long()
            ti = Variable(torch.from_numpy(np.array(t))).long()

            if self.cuda:
                xi = xi.cuda()
                yi = yi.cuda()
                ti = ti.cuda()

            bxs = [xi]
            bys = [yi]
            bts = [ti]

        else:
            bxs = []
            bys = []
            bts = []

        if len(self.M) > 0:
            order = [i for i in range(0,len(self.M))]
            osize = min(self.batchSize,len(self.M))
            for j in range(0,osize):
                shuffle(order)
                k = order[j]
                x,y,t = self.M[k]
                xi = Variable(torch.from_numpy(np.array(x))).float().unsqueeze(0) #.view(1,-1)
                yi = Variable(torch.from_numpy(np.array(y))).long()
                ti = Variable(torch.from_numpy(np.array(t))).long()

                # handle gpus if specified
                if self.cuda:
                    xi = xi.cuda()
                    yi = yi.cuda()
                    ti = ti.cuda()

                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        return bxs,bys,bts
               

    def observe(self, x, y, t):

        # step through elements of x
        for i in range(0,x.size()[0]):

            self.age += 1
            xi = x[i].data.cpu().numpy()
            yi = y[i].data.cpu().numpy()
            self.net.zero_grad()

            before = deepcopy(self.net.state_dict())
            for step in range(0,self.steps):
                weights_before = deepcopy(self.net.state_dict())
                ##Check for nan
                if weights_before != weights_before:
                    ipdb.set_trace()
                # Draw batch from buffer:
                bxs, bys, bts = self.getBatch(xi,yi,t)          
                loss = 0.0
                total_loss = 0.0
                for idx in range(len(bxs)):
                    
                    self.net.zero_grad()
                    bx = bxs[idx] 
                    by = bys[idx] 
                    bt = bts[idx]

                    if self.is_cifar:
                        offset1, offset2 = self.compute_offsets(bt)
                        prediction = (self.netforward(bx)[:, offset1:offset2])
                        loss = self.bce(prediction,
                                        by.unsqueeze(0)-offset1)
                    else:
                        prediction = self.forward(bx,0)
                        loss = self.bce(prediction, by.unsqueeze(0))
                    if torch.isnan(loss):
                        ipdb.set_trace()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                    self.opt.step()
                    total_loss += loss.item()
                weights_after = self.net.state_dict()
                if weights_after != weights_after:
                    ipdb.set_trace()

                # Within batch Reptile meta-update:
                self.net.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before})

            after = self.net.state_dict()

            # Across batch Reptile meta-update:
            self.net.load_state_dict({name : before[name] + ((after[name] - before[name]) * self.gamma) for name in before})

            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi,yi,t])

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = [xi,yi,t]

        return total_loss/self.steps

