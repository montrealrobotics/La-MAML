import random
import numpy as np
import ipdb
import math

import torch
import torch.nn as nn
from model.lamaml_base import *


class Net(BaseNet):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,           
                 args):
        super(Net, self).__init__(n_inputs,
                                 n_outputs,
                                 n_tasks,           
                                 args)

        self.nc_per_task = n_outputs

    def forward(self, x, t):
        output = self.net.forward(x)
        return output

    def meta_loss(self, x, fast_weights, y, t):
        """
        differentiate the loss through the network updates wrt alpha
        """
        logits = self.net.forward(x, fast_weights)
        loss_q = self.loss(logits.squeeze(1), y)
        return loss_q, logits

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        logits = self.net.forward(x, fast_weights)
        loss = self.loss(logits, y)   

        if fast_weights is None:
            fast_weights = self.net.parameters() 

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        grads = torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required)

        for i in range(len(grads)):
            torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        fast_weights = list(
                map(lambda p: p[1][0] - p[0] * nn.functional.relu(p[1][1]), zip(grads, zip(fast_weights, self.net.alpha_lr))))
        return fast_weights

    def observe(self, x, y, t):
        self.net.train() 

        for pass_itr in range(self.glances):
            self.pass_itr = pass_itr
            
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]
            
            self.epoch += 1
            self.zero_grads()

            if t != self.current_task:
                self.M = self.M_new
                self.current_task = t

            batch_sz = x.shape[0]
            meta_losses = [0 for _ in range(batch_sz)] 

            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)
            fast_weights = None

            for i in range(0, batch_sz):
                batch_x = x[i].unsqueeze(0)
                batch_y = y[i].unsqueeze(0)

                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)
                if(self.real_epoch == 0):
                    self.push_to_mem(batch_x, batch_y, torch.tensor(t))

                meta_loss, logits = self.meta_loss(bx, fast_weights, by, t) 
                meta_losses[i] += meta_loss
    
            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()

            meta_loss = sum(meta_losses)/len(meta_losses)

            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)

            if self.args.learn_lr:
                self.opt_lr.step()

            if(self.args.sync_update):
                self.opt_wt.step()
            else:  
                for i,p in enumerate(self.net.parameters()):
                    p.data = p.data - p.grad * nn.functional.relu(self.net.alpha_lr[i])
         
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        return meta_loss.item()