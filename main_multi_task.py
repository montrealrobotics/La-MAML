import time
import os
from tqdm import tqdm

import torch
from torch.autograd import Variable

def eval_iid_tasks(model, tasks, args):

    model.eval()
    result = []
    for t, task_loader in enumerate(tasks):
        rt = 0

        for (i, (x, y, super_y)) in enumerate(task_loader):
            if args.cuda:
                x = x.cuda()
            _, p = torch.max(model(x, super_y).data.cpu(), 1, keepdim=False)
            rt += (p == y).float().sum()

        result.append(rt / len(task_loader.dataset))
    return result

def life_experience_iid(model, inc_loader, args):
    result_val_a = []
    result_test_a = []

    result_val_t = []
    result_test_t = []

    time_start = time.time()
    test_tasks = inc_loader.get_tasks("test")
    val_tasks = inc_loader.get_tasks("val")
    task_info, train_loader, _, _ = inc_loader.new_task()
    evaluator = eval_iid_tasks

    for ep in range(args.n_epochs):

        model.real_epoch = ep

        prog_bar = tqdm(train_loader)
        for (i, (x, y, super_y)) in enumerate(prog_bar):
            if((i % args.log_every) == 0):
                result_val_a.append(evaluator(model, val_tasks, args))
                result_val_t.append(task_info["task"])

            v_x = x
            v_y = y
            if args.arch == 'linear':
                v_x = x.view(x.size(0), -1)
            super_v_y = super_y

            if args.cuda:
                v_x = v_x.cuda()
                v_y = v_y.cuda()
                super_v_y = super_v_y.cuda()

            model.train()

            loss = model.observe(Variable(v_x), Variable(v_y), Variable(super_v_y))

            prog_bar.set_description(
                "Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {}".format(
                    ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                    round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5)
                )
            )

    result_val_a.append(evaluator(model, val_tasks, args))
    result_val_t.append(task_info["task"])

    if args.calc_test_accuracy:
        result_test_a.append(evaluator(model, test_tasks, args))
        result_test_t.append(task_info["task"])


    print("####Final Validation Accuracy####")
    print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a[-1])/len(result_val_a[-1]), result_val_a[-1]))

    if args.calc_test_accuracy:
        print("####Final Test Accuracy####")
        print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_test_a[-1])/len(result_test_a[-1]), result_test_a[-1]))


    time_end = time.time()
    time_spent = time_end - time_start
    return torch.Tensor(result_val_t), torch.Tensor(result_val_a), torch.Tensor(result_test_t), torch.Tensor(result_test_a), time_spent

