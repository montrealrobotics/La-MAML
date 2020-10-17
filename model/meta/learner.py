import math
import os
import sys
import traceback
import numpy as np
import ipdb

import torch
from torch import nn
from torch.nn import functional as F

class Learner(nn.Module):

    def __init__(self, config, args = None):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config
        self.tf_counter = 0
        self.args = args

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        self.names = []

        for i, (name, param, extra_name) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]                
                if(self.args.xav_init):
                    w = nn.Parameter(torch.ones(*param[:4]))
                    b = nn.Parameter(torch.zeros(param[0]))
                    torch.nn.init.xavier_normal_(w.data)
                    b.data.normal_(0, math.sqrt(2)/math.sqrt(1+9*b.data.shape[0]))
                    self.vars.append(w)
                    self.vars.append(b)
                else:
                    w = nn.Parameter(torch.ones(*param[:4]))
                    # gain=1 according to cbfin's implementation
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    # [ch_out]
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # layer += 1
                if(self.args.xav_init):
                    w = nn.Parameter(torch.ones(*param))
                    # b = nn.Parameter(torch.zeros(param[0]))
                    torch.nn.init.xavier_normal_(w.data)
                    # b.data.normal_(0, math.sqrt(2)/math.sqrt(1+9*b.data.shape[0]))
                    self.vars.append(w)
                    # self.vars.append(b)
                else:     
                    # [ch_out, ch_in]
                    w = nn.Parameter(torch.ones(*param))
                    # gain=1 according to cbfinn's implementation
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'cat':
                pass
            elif name is 'cat_start':
                pass
            elif name is "rep":
                pass
            elif name in ["residual3", "residual5", "in"]:
                pass
            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):

        info = ''

        for name, param, extra_name in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name is 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name is 'rep':
                tmp = 'rep'
                info += tmp + "\n"


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=False, feature=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        cat_var = False
        cat_list = []

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        try:

            for (name, param, extra_name) in self.config:
                # assert(name == "conv2d")
                if name == 'conv2d':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2

                    # print(name, param, '\tout:', x.shape)
                elif name == 'convt2d':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2


                elif name == 'linear':

                    # ipdb.set_trace()
                    if extra_name == 'cosine':
                        w = F.normalize(vars[idx])
                        x = F.normalize(x)
                        x = F.linear(x, w)
                        idx += 1
                    else:
                        w, b = vars[idx], vars[idx + 1]
                        x = F.linear(x, w, b)
                        idx += 2

                    if cat_var:
                        cat_list.append(x)

                elif name == 'rep':
                    # print('rep')
                    # print(x.shape)
                    if feature:
                        return x

                elif name == "cat_start":
                    cat_var = True
                    cat_list = []

                elif name == "cat":
                    cat_var = False
                    x = torch.cat(cat_list, dim=1)

                elif name == 'bn':
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    idx += 2
                    bn_idx += 2
                elif name == 'flatten':
                    # print('flatten')
                    # print(x.shape)

                    x = x.view(x.size(0), -1)

                elif name == 'reshape':
                    # [b, 8] => [b, 2, 2, 2]
                    x = x.view(x.size(0), *param)
                elif name == 'relu':
                    x = F.relu(x, inplace=param[0])
                elif name == 'leakyrelu':
                    x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    x = F.tanh(x)
                elif name == 'sigmoid':
                    x = torch.sigmoid(x)
                elif name == 'upsample':
                    x = F.upsample_nearest(x, scale_factor=param[0])
                elif name == 'max_pool2d':
                    x = F.max_pool2d(x, param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    x = F.avg_pool2d(x, param[0], param[1], param[2])

                else:
                    print(name)
                    raise NotImplementedError

        except:
            traceback.print_exc(file=sys.stdout)
            ipdb.set_trace()

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def define_task_lr_params(self, alpha_init=1e-3): 
        # Setup learning parameters
        self.alpha_lr = nn.ParameterList([])

        self.lr_name = []
        for n, p in self.named_parameters():
            self.lr_name.append(n)

        for p in self.parameters():
            self.alpha_lr.append(nn.Parameter(alpha_init * torch.ones(p.shape, requires_grad=True)))                                           

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


