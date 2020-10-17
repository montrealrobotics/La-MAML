#!/bin/bash

ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_rotations    --cuda --log_dir logs/"
PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_permutations --cuda --log_dir logs/"
MANY="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 200 --dataset mnist_manypermutations --cuda --log_dir logs/"
CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
IMGNET='--data_path data/tiny-imagenet-200/ --log_every 100 --dataset tinyimagenet --cuda --log_dir logs/'
SEED=0

########## MNIST DATASETS ##########
##### La-MAML #####
# ROTATION
#cmaml
python3 main.py $ROT --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.1 \
                    --opt_wt 0.1 --alpha_init 0.1 --sync_update --use_old_task_memory --seed $SEED

#sync
python3 main.py $ROT --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.1 \
                    --opt_wt 0.3 --alpha_init 0.15 --learn_lr --sync_update --use_old_task_memory --seed $SEED

#lamaml
python3 main.py $ROT --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.3 \
                    --alpha_init 0.15 --learn_lr --use_old_task_memory --seed $SEED

#PERMUTATION
#cmaml
python3 main.py $PERM --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_wt 0.1 \
                    --alpha_init 0.03 --sync_update --use_old_task_memory --seed $SEED

#sync
python3 main.py $PERM --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.1 \
                    --opt_wt 0.1  --alpha_init 0.15 --learn_lr --sync_update --use_old_task_memory --seed $SEED

#lamaml
python3 main.py $PERM --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.3 \
                    --alpha_init 0.15 --learn_lr --use_old_task_memory --seed $SEED

#MANY
#cmaml
python3 main.py $MANY --model lamaml --memories 500 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_wt 0.15 \
                    --alpha_init 0.03 --sync_update --use_old_task_memory --seed $SEED

#sync
python3 main.py $MANY --model lamaml --memories 500 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 10 --opt_lr 0.1 \
                    --opt_wt 0.03  --alpha_init 0.03 --learn_lr --sync_update --use_old_task_memory --seed $SEED

#lamaml
python3 main.py $MANY --model lamaml --memories 500 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 10 --opt_lr 0.1 \
                    --alpha_init 0.1 --learn_lr --use_old_task_memory --seed $SEED

########## CIFAR DATASET Multi-Pass ##########
##### IID ##### 
python3 main.py $CIFAR --model iid2 --expt_name iid2 --batch_size 32 --n_epochs 50 \
                    --lr 0.03 --loader multi_task_loader --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random --test_batch_size 1280  --samples_per_task 2500 \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### ER ##### 
python3 main.py $CIFAR --model eralg4 --expt_name eralg4 --memories 200 --batch_size 10 --replay_batch_size 1 --n_epochs 10 \
                     --lr 0.03 --glances 1 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### La-ER ##### 
python3 main.py $CIFAR --model eralg4 --expt_name eralg4 --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.1 --alpha_init 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn"  --cifar_batches 5 --learn_lr --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### Icarl ##### 
python3 main.py $CIFAR --model icarl --expt_name icarl --n_memories 200 --batch_size 10 --n_epochs 10 \
                     --lr 0.03 --glances 1 --memory_strength 1.0 --loader class_incremental_loader --increment 5 \
                     --arch "pc_cnn" --log_every 3125 --class_order random  --samples_per_task 2500 \
                     --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.1

##### GEM ##### 
python3 main.py $CIFAR --model gem --expt_name gem --n_memories 10 --batch_size 10 --n_epochs 10 \
                    --lr 0.03 --glances 1 --memory_strength 0.5 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random --samples_per_task 2500 \
                    --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.1

##### AGEM ##### 
python3 main.py $CIFAR --model agem --expt_name agem --n_memories 10 --batch_size 10 --n_epochs 10 \
                    --lr 0.03 --glances 1 --memory_strength 0.5 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### Meta BGD ##### 
python3 main.py $CIFAR --model meta-bgd --expt_name meta-bgd --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --alpha_init 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 3 --log_every 3125 --second_order --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1 --xav_init  --std_init 0.02 --mean_eta 50. --train_mc_iters 2

##### sync ##### 
python3 main.py $CIFAR --model lamaml_cifar --expt_name lamaml_sync --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.35 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --sync_update --log_every 3125 --second_order --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### C-MAML #####
python3 main.py $CIFAR --model lamaml_cifar --expt_name cmaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.35 --alpha_init 0.075 --opt_wt 0.075 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --sync_update --log_every 3125 --second_order --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### La-MAML #####
python3 main.py $CIFAR --model lamaml_cifar --expt_name lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.25 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1


########## CIFAR DATASET Single-Pass ##########
##### ER ##### 
python3 main.py $CIFAR --model eralg4 --expt_name eralg4 --memories 200 --batch_size 10 --replay_batch_size 1 --n_epochs 1 \
                     --lr 0.03 --glances 10 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### La-ER ##### 
python3 main.py $CIFAR --model eralg4 --expt_name eralg4 --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --opt_lr 0.1 --alpha_init 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn"  --cifar_batches 5 --learn_lr --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### Icarl ##### 
python3 main.py $CIFAR --model icarl --expt_name icarl --n_memories 200 --batch_size 10 --n_epochs 1 \
                     --lr 0.03 --glances 2 --memory_strength 1.0 --loader class_incremental_loader --increment 5 \
                     --arch "pc_cnn" --log_every 3125 --class_order random  --samples_per_task 2500 \
                     --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.1

##### GEM ##### 
python3 main.py $CIFAR --model gem --expt_name gem --n_memories 10 --batch_size 10 --n_epochs 1 \
                    --lr 0.03 --glances 2 --memory_strength 0.5 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random --samples_per_task 2500 \
                    --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.1

##### AGEM ##### 
python3 main.py $CIFAR --model agem --expt_name agem --n_memories 10 --batch_size 10 --n_epochs 1 \
                    --lr 0.03 --glances 2 --memory_strength 0.5 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### MER ##### 
python3 main.py $CIFAR --model meralg1 --expt_name meralg1 --memories 200 --replay_batch_size 25 \
                    --lr 0.1 --beta 0.1 --gamma 1.0 --batches_per_example 10 --loader class_incremental_loader  --increment 5 \
                    --arch 'pc_cnn' --log_every 3125 --grad_clip_norm 10.0 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### Meta BGD ##### 
python3 main.py $CIFAR --model meta-bgd --expt_name meta-bgd --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --alpha_init 0.1 --glances 3 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 3 --log_every 3125 --second_order --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1 --xav_init  --std_init 0.02 --mean_eta 50. --train_mc_iters 2

##### sync ##### 
python3 main.py $CIFAR --model lamaml_cifar --expt_name lamaml_sync --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --opt_lr 0.35 --alpha_init 0.1 --opt_wt 0.1 --glances 5 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --sync_update --log_every 3125 --second_order --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### C-MAML #####
python3 main.py $CIFAR --model lamaml_cifar --expt_name cmaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --opt_lr 0.35 --alpha_init 0.075 --opt_wt 0.075 --glances 2 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --sync_update --log_every 3125 --second_order --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

##### La-MAML #####
python3 main.py $CIFAR --model lamaml_cifar --expt_name lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --opt_lr 0.25 --alpha_init 0.1 --opt_wt 0.1 --glances 10 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1

