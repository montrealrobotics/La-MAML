import torch.optim as optim
from .bgd_optimizer import BGD


def bgd(model, **kwargs):
    # logger = kwargs.get("logger", None)
    # assert(logger is not None)
    bgd_params = {
        "mean_eta": kwargs.get("mean_eta", 1),
        "std_init": kwargs.get("std_init", 0.02),
        "mc_iters": kwargs.get("mc_iters", 10)
    }
    # logger.info("BGD params: " + str(bgd_params))
    all_params = [{'params': params} for l, (name, params) in enumerate(model.named_parameters())]
    return BGD(all_params, **bgd_params)


def sgd(model, **kwargs):
    # logger = kwargs.get("logger", None)
    # assert(logger is not None)
    sgd_params = {
        "momentum": kwargs.get("momentum", 0.9),
        "lr": kwargs.get("lr", 0.1),
        "weight_decay": kwargs.get("weight_decay", 5e-4)
    }
    # logger.info("SGD params: " + str(sgd_params))
    all_params = [{'params': params, 'name': name, 'initial_lr': kwargs.get("lr", 0.1)} for l, (name, params) in enumerate(model.named_parameters())]
    return optim.SGD(all_params, **sgd_params)


def adam(model, **kwargs):
    # logger = kwargs.get("logger", None)
    # assert(logger is not None)
    adam_params = {
        "eps": kwargs.get("eps", 1e-08),
        "lr": kwargs.get("lr", 0.001),
        "betas": kwargs.get("betas", (0.9, 0.999)),
        "weight_decay": kwargs.get("weight_decay", 0)
    }
    # logger.info("ADAM params: " + str(adam_params))
    all_params = [{'params': params, 'name': name, 'initial_lr': kwargs.get("lr", 0.001)} for l, (name, params) in enumerate(model.named_parameters())]
    return optim.Adam(all_params, **adam_params)


def adagrad(model, **kwargs):
    # logger = kwargs.get("logger", None)
    # assert(logger is not None)
    adam_params = {
        "lr": kwargs.get("lr", 0.01),
        "weight_decay": kwargs.get("weight_decay", 0)
    }
    # logger.info("Adagrad params: " + str(adam_params))
    all_params = [{'params': params, 'name': name, 'initial_lr': kwargs.get("lr", 0.01)} for l, (name, params) in enumerate(model.named_parameters())]
    return optim.Adagrad(all_params, **adam_params)
