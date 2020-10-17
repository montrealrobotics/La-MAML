import random
from random import shuffle
import numpy as np
import ipdb
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import model.meta.learner as Learner
import model.meta.modelfactory as mf

from model.optimizers_lib import optimizers_lib
from ast import literal_eval

"""
This baseline/ablation is constructed by merging C-MAML and BGD
By assigning a variance parameter to each NN parameter in the model
and using BGD's bayesian update to update these means (the NN parameters) and variances
(the learning rates in BGD are derived from the variances)

The 'n' bayesian samples in this case are the 'n' cumulative meta-losses sampled when 
C-MAML is run with 'n' different initial theta vectors as the NN means sampled from the 
(means, variances) stored for the model parameters.
The weight update is then carried out using the BGD formula that implicitly 
uses the variances to derive the learning rates for the parameters
"""

class Net(torch.nn.Module):

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

        # define the lr params
        self.net.define_task_lr_params(alpha_init = args.alpha_init)

        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

        # optimizer model
        optimizer_model = optimizers_lib.__dict__[args.bgd_optimizer]
        # params used to instantiate the BGD optimiser
        optimizer_params = dict({ #"logger": logger,
                                 "mean_eta": args.mean_eta,
                                 "std_init": args.std_init,
                                 "mc_iters": args.train_mc_iters}, **literal_eval(" ".join(args.optimizer_params)))
        self.optimizer = optimizer_model(self.net, **optimizer_params)

        self.epoch = 0
        # allocate buffer
        self.M = []        
        self.M_new = []
        self.age = 0

        # setup losses
        self.loss = torch.nn.CrossEntropyLoss()
        self.is_cifar = ((args.dataset == 'cifar100') or (args.dataset == 'tinyimagenet'))
        self.glances = args.glances
        self.pass_itr = 0
        self.real_epoch = 0

        # setup memories
        self.current_task = 0

        self.memories = args.memories
        self.batchSize = int(args.replay_batch_size)

        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs

        self.obseve_itr = 0

    def take_multitask_loss(self, bt, t, logits, y):
        loss = 0.0

        for i, ti in enumerate(bt):
            offset1, offset2 = self.compute_offsets(ti)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)


    def forward(self, x, t, fast_weights=None):
        self.optimizer.randomize_weights(force_std=0)  
        output = self.net.forward(x, vars=fast_weights)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def meta_loss(self, x, fast_weights, y, bt, t):
        """
        differentiate the loss through the network updates wrt alpha
        """

        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            logits = self.net.forward(x, fast_weights)[:, :offset2]                   

            loss_q = self.take_multitask_loss(bt, t, logits, y)
        else:
            logits = self.net.forward(x, fast_weights)
            # Cross Entropy Loss over data
            loss_q = self.loss(logits, y)
        return loss_q, logits

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def push_to_mem(self, batch_x, batch_y, t):
        """
        Reservoir sampling memory update
        """

        if(self.real_epoch > 0 or self.pass_itr>0):
            return
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()              
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M_new) < self.memories:
                self.M_new.append([batch_x[i], batch_y[i], t])
            else:
                p = random.randint(0,self.age)  
                if p < self.memories:
                    self.M_new[p] = [batch_x[i], batch_y[i], t]


    def getBatch(self, x, y, t):
        """
        Given the new data points, create a batch of old + new data, 
        where old data is part of the memory buffer
        """

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

        if self.args.use_old_task_memory: # and t>0:
            MEM = self.M
        else:
            MEM = self.M_new
        
        if len(MEM) > 0:
            order = [i for i in range(0,len(MEM))]
            osize = min(self.batchSize,len(MEM))
            for j in range(0,osize):
                shuffle(order)
                k = order[j]
                x,y,t = MEM[k]

                xi = np.array(x)
                yi = np.array(y)
                ti = np.array(t)
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        for j in range(len(myi)):
            bxs.append(mxi[j])
            bys.append(myi[j])
            bts.append(mti[j])

        bxs = Variable(torch.from_numpy(np.array(bxs))).float()
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)
        
        # handle gpus if specified
        if self.cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs,bys,bts

    def take_loss(self, t, logits, y):
        offset1, offset2 = self.compute_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y-offset1)

        return loss

    def inner_update(self, x, fast_weights, y, t):            
        """
        Update the fast weights using the current samples and return the updated fast
        """
        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            logits = self.net.forward(x, fast_weights)[:, :offset2]

            loss = self.take_loss(t, logits, y)
            # loss = self.loss(logits, y)
        else:
            logits = self.net.forward(x, fast_weights)
            loss = self.loss(logits, y)   

        if fast_weights is None:
            fast_weights = self.net.parameters()      

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = True
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required))
        
        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)            

        # get fast weights vector by taking SGD step on grads
        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.net.alpha_lr))))
        return fast_weights


    def observe(self, x, y, t):
        self.net.train()             
        self.obseve_itr += 1
                                                
        num_of_mc_iters = self.optimizer.get_mc_iters()

        for glance_itr in range(self.glances):

            mc_meta_losses = [0 for _ in range(num_of_mc_iters)]

            # running C-MAML num_of_mc_iters times to get montecarlo samples of meta-loss
            for pass_itr in range(num_of_mc_iters):
                self.optimizer.randomize_weights()                                                          

                self.pass_itr = pass_itr
                self.epoch += 1
                self.net.zero_grad()                      

                perm = torch.randperm(x.size(0))
                x = x[perm]
                y = y[perm]


                if pass_itr==0 and glance_itr ==0 and t != self.current_task:
                    self.M = self.M_new
                    self.current_task = t

                batch_sz = x.shape[0]

                n_batches = self.args.cifar_batches
                rough_sz = math.ceil(batch_sz/n_batches)

                # the samples of new task to iterate over in inner update trajectory
                iterate_till = 1 #batch_sz 
                meta_losses = [0 for _ in range(n_batches)]  
                accuracy_meta_set = [0 for _ in range(n_batches)] 

                # put some asserts to make sure replay batch size can accomodate old and new samples
                bx, by = None, None
                bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)             
                    
                fast_weights = None
                # inner loop/fast updates where learn on 1-2 samples in each inner step
                for i in range(n_batches):

                    batch_x = x[i*rough_sz : (i+1)*rough_sz]
                    batch_y = y[i*rough_sz : (i+1)*rough_sz]
                    fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)

                    if(pass_itr==0 and glance_itr==0):
                        self.push_to_mem(batch_x, batch_y, torch.tensor(t))

                    # the meta loss is computed at each inner step
                    # as this is shown to work better in Reptile [] 
                    meta_loss, logits = self.meta_loss(bx, fast_weights, by, bt, t) 
                    meta_losses[i] += meta_loss

                self.optimizer.zero_grad()                                                                   
                meta_loss = sum(meta_losses)/len(meta_losses)
                if torch.isnan(meta_loss):
                    ipdb.set_trace()
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                mc_meta_losses[pass_itr] = meta_loss
                self.optimizer.aggregate_grads(batch_size=batch_sz)           
             
            print_std = False                        
            if(self.obseve_itr%220==0):
                print_std = True                                                         
            self.optimizer.step(print_std = print_std)

        meta_loss_return = sum(mc_meta_losses)/len(mc_meta_losses)

        return meta_loss_return.item()

