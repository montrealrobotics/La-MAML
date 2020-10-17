# An implementation of Experience Replay (ER) with reservoir sampling and without using tasks from Algorithm 4 of https://openreview.net/pdf?id=B1gTShAct7

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
import warnings
import math

import model.meta.modelfactory as mf
import model.meta.learner as Learner
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

        config = mf.ModelFactory.get_model(model_type = args.arch, sizes = [n_inputs] + [nh] * nl + [n_outputs],
                                                dataset = args.dataset, args=args)
        self.net = Learner.Learner(config, args)

        self.opt_wt = optim.SGD(self.parameters(), lr=args.lr)

        if self.args.learn_lr:
            self.net.define_task_lr_params(alpha_init = args.alpha_init)
            self.opt_lr = torch.optim.SGD(list(self.net.alpha_lr.parameters()), lr=args.opt_lr)          

        self.loss = CrossEntropyLoss()
        self.is_cifar = ((args.dataset == 'cifar100') or (args.dataset == 'tinyimagenet'))
        self.glances = args.glances

        self.current_task = 0
        self.memories = args.memories
        self.batchSize = int(args.replay_batch_size)

        # allocate buffer
        self.M = []
        self.age = 0
        
        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

        self.n_outputs = n_outputs
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs


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
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def getBatch(self, x, y, t):
        if(x is not None):
            mxi = np.array(x)
            myi = np.array(y)
            mti = np.ones(x.shape[0], dtype=int)*t            
        else:
            mxi = np.empty( shape=(0, 0) )
            myi = np.empty( shape=(0, 0) )
            mti = np.empty( shape=(0, 0) )

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
                xi = np.array(x)
                yi = np.array(y)
                ti = np.array(t)
                
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        for i in range(len(myi)):
            bxs.append(mxi[i])
            bys.append(myi[i])
            bts.append(mti[i])

        bxs = Variable(torch.from_numpy(np.array(bxs))).float()
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)
        
        # handle gpus if specified
        if self.cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()
 
        return bxs,bys,bts


    def observe(self, x, y, t):
        ### step through elements of x

        xi = x.data.cpu().numpy()
        yi = y.data.cpu().numpy()

        if t != self.current_task:
           self.current_task = t

        if self.args.learn_lr:
            loss = self.la_ER(x, y, t)
        else:
            loss = self.ER(xi, yi, t)

        for i in range(0, x.size()[0]):
            self.age += 1
            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi[i], yi[i], t])

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = [xi[i], yi[i], t]

        return loss.item()

    def ER(self, x, y, t):
        for pass_itr in range(self.glances):

            self.net.zero_grad()
            
            # Draw batch from buffer:
            bx,by,bt = self.getBatch(x,y,t)

            bx = bx.squeeze()
            prediction = self.net.forward(bx)
            loss = self.take_multitask_loss(bt, prediction, by)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

            self.opt_wt.step()
        
        return loss

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """

        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)            
            logits = self.net.forward(x, fast_weights)[:, :offset2]
            loss = self.loss(logits[:, offset1:offset2], y-offset1)
        else:
            logits = self.net.forward(x, fast_weights)
            loss = self.loss(logits, y)   

        if fast_weights is None:
            fast_weights = self.net.parameters()

        graph_required = self.args.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required))

        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.net.alpha_lr))))
        return fast_weights, loss.item()


    def la_ER(self, x, y, t):
        """
        this ablation tests whether it suffices to just do the learning rate modulation
        guided by gradient alignment + clipping (that La-MAML does implciitly through autodiff)
        and use it with ER (therefore no meta-learning for the weights)

        """
        for pass_itr in range(self.glances):
            
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            batch_sz = x.shape[0]
            n_batches = self.args.cifar_batches
            rough_sz = math.ceil(batch_sz/n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)] 

            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)
            bx = bx.squeeze()
            
            for i in range(n_batches):

                batch_x = x[i*rough_sz : (i+1)*rough_sz]
                batch_y = y[i*rough_sz : (i+1)*rough_sz]

                # assuming labels for inner update are from the same 
                fast_weights, inner_loss = self.inner_update(batch_x, fast_weights, batch_y, t)

                prediction = self.net.forward(bx, fast_weights)
                meta_loss = self.take_multitask_loss(bt, prediction, by)
                meta_losses[i] += meta_loss

            # update alphas
            self.net.zero_grad()
            self.opt_lr.zero_grad()

            meta_loss = meta_losses[-1] #sum(meta_losses)/len(meta_losses)
            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)
            
            # update the LRs (guided by meta-loss, but not the weights)
            self.opt_lr.step()

            # update weights
            self.net.zero_grad()

            # compute ER loss for network weights
            prediction = self.net.forward(bx)
            loss = self.take_multitask_loss(bt, prediction, by)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

            # update weights with grad from simple ER loss 
            # and LRs obtained from meta-loss guided by old and new tasks
            for i,p in enumerate(self.net.parameters()):                                 
                p.data = p.data - (p.grad * nn.functional.relu(self.net.alpha_lr[i]))       
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        return loss