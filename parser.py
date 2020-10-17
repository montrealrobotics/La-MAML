# coding=utf-8
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Continual learning')
    parser.add_argument('--expt_name', type=str, default='test_lamaml',
                    help='name of the experiment')
    
    # model details
    parser.add_argument('--model', type=str, default='single',
                        help='algo to train')
    parser.add_argument('--arch', type=str, default='linear', 
                        help='arch to use for training', choices = ['linear', 'pc_cnn'])
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--xav_init', default=False , action='store_true',
                        help='Use xavier initialization')



    # optimizer parameters influencing all models
    parser.add_argument("--glances", default=1, type=int,
                        help="Number of times the model is allowed to train over a set of samples in the single pass setting") 
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the amount of items received by the algorithm at one time (set to 1 across all ' +
                        'experiments). Variable name is from GEM project.')
    parser.add_argument('--replay_batch_size', type=float, default=20,
                        help='The batch size for experience replay.')
    parser.add_argument('--memories', type=int, default=5120, 
                        help='number of total memories stored in a reservoir sampling based buffer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (For baselines)')

    
    # experiment parameters
    parser.add_argument('--cuda', default=False , action='store_true',
                        help='Use GPU')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--log_every', type=int, default=1000,
                        help='frequency of checking the validation accuracy, in minibatches')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='the directory where the logs will be saved')
    parser.add_argument('--tf_dir', type=str, default='',
                        help='(not set by user)')
    parser.add_argument('--calc_test_accuracy', default=False , action='store_true',
                        help='Calculate test accuracy along with val accuracy')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--loader', type=str, default='task_incremental_loader',
                        help='data loader to use')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', default=False, action='store_true',
                        help='present tasks in order')
    parser.add_argument('--classes_per_it', type=int, default=4,
                        help='number of classes in every batch')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='number of classes in every batch')
    parser.add_argument("--dataset", default="mnist_rotations", type=str,
                    help="Dataset to train and test on.")
    parser.add_argument("--workers", default=3, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("--validation", default=0., type=float,
                        help="Validation split (0. <= x <= 1.).")
    parser.add_argument("-order", "--class_order", default="old", type=str,
                        help="define classes order of increment ",
                        choices = ["random", "chrono", "old", "super"])
    parser.add_argument("-inc", "--increment", default=5, type=int,
                        help="number of classes to increment by in class incremental loader")
    parser.add_argument('--test_batch_size', type=int, default=100000 ,
                        help='batch size to use during testing.')


    # La-MAML parameters
    parser.add_argument('--opt_lr', type=float, default=1e-1,
                        help='learning rate for LRs')
    parser.add_argument('--opt_wt', type=float, default=1e-1,
                        help='learning rate for weights')
    parser.add_argument('--alpha_init', type=float, default=1e-3,
                        help='initialization for the LRs')
    parser.add_argument('--learn_lr', default=False, action='store_true',
                        help='model should update the LRs during learning')
    parser.add_argument('--sync_update', default=False , action='store_true',
                        help='the LRs and weights should be updated synchronously')

    parser.add_argument('--grad_clip_norm', type=float, default=2.0,
                        help='Clip the gradients by this value')
    parser.add_argument("--cifar_batches", default=3, type=int,
                        help="Number of batches in inner trajectory") 
    parser.add_argument('--use_old_task_memory', default=False, action='store_true', 
                        help='Use only old task samples for replay buffer data')    
    parser.add_argument('--second_order', default=False , action='store_true',
                        help='use second order MAML updates')


   # memory parameters for GEM | AGEM | ICARL 
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--steps_per_sample', default=1, type=int,
                        help='training steps per batch')


    # parameters specific to MER 
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma learning rate parameter')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta learning rate parameter')
    parser.add_argument('--s', type=float, default=1,
                        help='current example learning rate multiplier (s)')
    parser.add_argument('--batches_per_example', type=float, default=1,
                        help='the number of batch per incoming example')


    # parameters specific to Meta-BGD
    parser.add_argument('--bgd_optimizer', type=str, default="bgd", choices=["adam", "adagrad", "bgd", "sgd"],
                    help='Optimizer.')
    parser.add_argument('--optimizer_params', default="{}", type=str, nargs='*',
                        help='Optimizer parameters')

    parser.add_argument('--train_mc_iters', default=5, type=int,
                        help='Number of MonteCarlo samples during training(default 10)')
    parser.add_argument('--std_init', default=5e-2, type=float,
                        help='STD init value (default 5e-2)')
    parser.add_argument('--mean_eta', default=1, type=float,
                        help='Eta for mean step (default 1)')
    parser.add_argument('--fisher_gamma', default=0.95, type=float,
                        help='')

    return parser