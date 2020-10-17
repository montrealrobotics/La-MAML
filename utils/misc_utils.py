import datetime
import glob
import json
import os
import random
import ipdb
import numpy as np
import torch
from tqdm import tqdm



def to_onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot

def _check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss >= 0.).item())

def compute_accuracy(ypred, ytrue, task_size=10):
    all_acc = {}

    all_acc["total"] = round((ypred == ytrue).sum() / len(ytrue), 3)

    for class_id in range(0, np.max(ytrue), task_size):
        idxes = np.where(
                np.logical_and(ytrue >= class_id, ytrue < class_id + task_size)
        )[0]

        label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
        )
        all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

    return all_acc


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")


def get_date_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-2]


def log_dir(opt, timestamp=None):
    if timestamp is None:
        timestamp = get_date_time()

    rand_num = str(random.randint(1,1001))
    logdir = opt.log_dir + '/%s/%s-%s/%s' % (opt.model, opt.expt_name, timestamp, opt.seed)
    tfdir = opt.log_dir +  '/%s/%s-%s/%s/%s' % (opt.model, opt.expt_name, timestamp, opt.seed, "tfdir")

    mkdir(logdir)
    mkdir(tfdir)
    
    with open(logdir + '/training_parameters.json', 'w') as f:
        json.dump(vars(opt), f, indent=4)
    
    return logdir, tfdir


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def find_latest_checkpoint(folder_path):
    print('searching for checkpoint in : '+folder_path)
    files = sorted(glob.iglob(folder_path+'/*.pth'), key=os.path.getmtime, reverse=True)
    print('latest checkpoint is:')
    print(files[0])
    return files[0]


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    print("Set seed", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.enabled = False


def find_latest_checkpoint_name(folder_path):
    print('searching for checkpoint in : '+folder_path)
    files = glob.glob(folder_path+'/*.pth')
    min_num = 0
    filename = ''
    for i, filei in enumerate(files):
        ckpt_name = os.path.splitext(filei) 
        ckpt_num = int(ckpt_name.split('_')[-1])
        if(ckpt_num>min_num):
            min_num = ckpt_num
            filename = filei
    print('latest checkpoint is:')
    print(filename)
    return filename


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))


def log_sum_exp(input, dim=None, keepdim=False):
    """Numerically stable LogSumExp.

    Args:
        input (Tensor)
        dim (int): Dimension along with the sum is performed
        keepdim (bool): Whether to retain the last dimension on summing

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        input = input.view(-1)
        dim = 0
    max_val = input.max(dim=dim, keepdim=True)[0]
    output = max_val + (input - max_val).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        output = output.squeeze(dim)
    return output