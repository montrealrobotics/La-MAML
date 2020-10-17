import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from dataloaders.idataset import DummyArrayDataset
import os


class IncrementalLoader:

    def __init__(
        self,
        opt,
        shuffle=True,
        seed=1,
    ):
        self._opt = opt
        validation_split=opt.validation
        increment=opt.increment

        self._setup_data(
            class_order_type=opt.class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )

        self._current_task = 0

        self._batch_size = opt.batch_size
        self._test_batch_size = opt.test_batch_size        
        self._workers = opt.workers
        self._shuffle = shuffle

        self._setup_test_tasks()

    @property
    def n_tasks(self):
        return len(self.test_dataset)
    
    def new_task(self):
        if self._current_task >= len(self.test_dataset):
            raise Exception("No more tasks.")

        p = self.sample_permutations[self._current_task]
        x_train, y_train = self.train_dataset[self._current_task][1][p], self.train_dataset[self._current_task][2][p]
        x_test, y_test = self.test_dataset[self._current_task][1], self.test_dataset[self._current_task][2]

        train_loader = self._get_loader(x_train, y_train, mode="train")
        test_loader = self._get_loader(x_test, y_test, mode="test")

        task_info = {
            "min_class": 0,
            "max_class": self.n_outputs,
            "increment": -1,
            "task": self._current_task,
            "max_task": len(self.test_dataset),
            "n_train_data": len(x_train),
            "n_test_data": len(x_test)
        }

        self._current_task += 1

        return task_info, train_loader, None, test_loader

    def _setup_test_tasks(self):
        self.test_tasks = []
        for i in range(len(self.test_dataset)):
            self.test_tasks.append(self._get_loader(self.test_dataset[i][1], self.test_dataset[i][2], mode="test"))

    def get_tasks(self, dataset_type='test'):
        if dataset_type == 'test':
            return self.test_dataset
        elif dataset_type == 'val':
            return self.test_dataset
        else:
            raise NotImplementedError("Unknown mode {}.".format(dataset_type))

    def get_dataset_info(self):
        n_inputs = self.train_dataset[0][1].size(1)
        n_outputs = 0
        for i in range(len(self.train_dataset)):
            n_outputs = max(n_outputs, self.train_dataset[i][2].max())
            n_outputs = max(n_outputs, self.test_dataset[i][2].max())
        self.n_outputs = n_outputs
        return n_inputs, n_outputs.item()+1, self.n_tasks


    def _get_loader(self, x, y, shuffle=True, mode="train"):
        if mode == "train":
            batch_size = self._batch_size
        elif mode == "test":
            batch_size = self._test_batch_size
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        return DataLoader(
            DummyArrayDataset(x, y),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._workers
        )


    def _setup_data(self, class_order_type=False, seed=1, increment=10, validation_split=0.):
        # FIXME: handles online loading of images
        torch.manual_seed(seed)

        self.train_dataset, self.test_dataset = torch.load(os.path.join(self._opt.data_path, self._opt.dataset + ".pt"))

        self.sample_permutations = []

        # for every task, accumulate a shuffled set of samples_per_task
        for t in range(len(self.train_dataset)):
            N = self.train_dataset[t][1].size(0)
            if self._opt.samples_per_task <= 0:
                n = N
            else:
                n = min(self._opt.samples_per_task, N)


            p = torch.randperm(N)[0:n]
            self.sample_permutations.append(p)
