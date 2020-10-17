import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from dataloaders.idataset import _get_datasets, DummyDataset
from dataloaders.task_sampler import MultiTaskSampler

import random
import ipdb


class IncrementalLoader:

    def __init__(
        self,
        opt,
        shuffle=True,
        seed=1,
    ):
        self._opt = opt
        dataset_name=opt.dataset
        validation_split=opt.validation
        self.increment=opt.increment

        datasets = _get_datasets(dataset_name)
        self._setup_data(
            datasets,
            class_order_type=opt.class_order,
            seed=seed,
            increment=self.increment,
            validation_split=validation_split
        )

        self.validation_split = validation_split
        self.train_transforms = datasets[0].train_transforms
        self.common_transforms = datasets[0].common_transforms
        self.top_transforms = datasets[0].top_transforms

        self._current_task = 0

        self._batch_size = opt.batch_size
        self._test_batch_size = opt.test_batch_size        
        self._workers = opt.workers
        self._shuffle = shuffle

        self._setup_test_tasks(validation_split)

    @property
    def n_tasks(self):
        return len(self.increments)
    
    def new_task(self):
        min_class = 0
        max_class = max(self.targets_train) + 1
        x_train, y_train, super_y_train = self._select(
            self.data_train, self.targets_train, self.super_targets_train, low_range=min_class, high_range=max_class
        )
        x_val, y_val, super_y_val = self._select(
            self.data_val, self.targets_val, self.super_targets_val, low_range=min_class, high_range=max_class
        )
        x_test, y_test, super_y_test = self._select(self.data_test, self.targets_test, self.super_targets_test, high_range=max_class)

        train_loader = self._get_loader(x_train, y_train, super_y_train, mode="train")
        val_loader = self._get_loader(x_val, y_val, super_y_val, mode="train") if len(x_val) > 0 else None
        test_loader = self._get_loader(x_test, y_test, super_y_test, mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0]
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    def _setup_test_tasks(self, validation_split):
        self.test_tasks = []
        self.val_tasks = []
        for i in range(len(self.increments)):
            min_class = i
            max_class = i+1
            x_test, y_test, super_y_test = self._select_super(self.data_test, self.targets_test, self.super_targets_test, low_range=min_class, high_range=max_class)
            self.test_tasks.append(self._get_loader(x_test, y_test, super_y_test, mode="test"))

            if validation_split > 0.0:
                x_val, y_val, super_y_val = self._select_super(self.data_val, self.targets_val, self.super_targets_val, low_range=min_class, high_range=max_class)
                self.val_tasks.append(self._get_loader(x_val, y_val, super_y_val, mode="test"))


    def get_tasks(self, dataset_type='test'):
        if dataset_type == 'val':
            if self.validation_split > 0.0:
                return self.val_tasks
            else:
                return self.test_tasks
        elif dataset_type == 'test':
            return self.test_tasks
        else:
            raise NotImplementedError("Unknown mode {}.".format(dataset_type))

    def get_dataset_info(self):
        if(self._opt.dataset == 'tinyimagenet'):
            n_inputs = 3*64*64        
        else:
            n_inputs = self.data_train.shape[3]*self.data_train.shape[1]*self.data_train.shape[2]
        n_outputs = self._opt.increment * len(self.increments)
        n_task = len(self.increments)
        return n_inputs, n_outputs, n_task

    def _select(self, x, y, super_y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes], super_y[idxes]

    def _select_super(self, x, y, super_y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(super_y >= low_range, super_y < high_range))[0]
        return x[idxes], y[idxes], super_y[idxes]

    def _get_loader(self, x, y, super_y, shuffle=True, mode="train"):
        if mode == "train":
            pretrsf = transforms.Compose([*self.top_transforms])
            trsf = transforms.Compose([*self.train_transforms, *self.common_transforms])
            batch_size = self._batch_size

            sampler = self._get_sampler(super_y, mode)

            return DataLoader(
                DummyDataset(x, y, trsf, pretrsf, self._opt.dataset=='tinyimagenet', super_y),
                batch_sampler=sampler,
                shuffle=False,
                num_workers=self._workers
            )
        elif mode == "test" or mode == "flip":
            if mode == "test":
                pretrsf = transforms.Compose([*self.top_transforms])
                trsf = transforms.Compose(self.common_transforms)
                batch_size = self._test_batch_size
            elif mode == "flip":
                trsf = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=1.), *self.common_transforms]
                )
                batch_size = self._test_batch_size

            return DataLoader(
                DummyDataset(x, y, trsf, pretrsf, self._opt.dataset=='tinyimagenet', super_y),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self._workers
            )
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))


    def _get_sampler(self, labels, mode):
        assert self._batch_size%self._opt.classes_per_it == 0, \
            "Batch size should be a multiple of number of desired classes in a iter"
        if 'train' in mode:
            classes_per_it = self._opt.classes_per_it
            num_samples = int(self._batch_size/self._opt.classes_per_it)
        elif 'val' in mode:
            classes_per_it = self._opt.classes_per_it
            num_samples = int(self._batch_size/self._opt.classes_per_it)
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        return MultiTaskSampler(labels=labels,
                                classes_per_it=classes_per_it,
                                num_samples=num_samples,
                                iterations=self._opt.iterations)

    def _setup_data(self, datasets, class_order_type=False, seed=1, increment=10, validation_split=0.):
        # FIXME: handles online loading of images
        self.data_train, self.targets_train, self.super_targets_train = [], [], []
        self.data_test, self.targets_test, self.super_targets_test = [], [], []
        self.data_val, self.targets_val, self.super_targets_val = [], [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:

            if(self._opt.dataset == 'tinyimagenet'):
                root_path = self._opt.data_path
                train_dataset = dataset.base_dataset(root_path + 'train/')
                test_dataset = dataset.base_dataset(root_path + 'val/')

                train_dataset.data = train_dataset.samples
                test_dataset.data = test_dataset.samples

                x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
                x_val, y_val, x_train, y_train = self._list_split_per_class(
                    x_train, y_train
                )
                x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

                order = [i for i in range(len(np.unique(y_train)))]
                if class_order_type == 'random':
                    random.seed(seed)  # Ensure that following order is determined by seed:
                    random.shuffle(order)
                    print("Class order:", order)
                elif class_order_type == 'old' and dataset.class_order is not None:
                    order = dataset.class_order
                else:
                    print("Classes are presented in a chronological order")

                self.class_order.append(order)

                y_train = self._map_new_class_index(y_train, order)
                y_val = self._map_new_class_index(y_val, order)
                y_test = self._map_new_class_index(y_test, order)

                super_y_train = self._make_super_classes(y_train, self.increment)
                super_y_test = self._make_super_classes(y_test, self.increment)
                super_y_val = self._make_super_classes(y_val, self.increment)

                y_train += current_class_idx
                y_val += current_class_idx
                y_test += current_class_idx

                current_class_idx += len(order)
                if len(datasets) > 1:
                    self.increments.append(len(order))
                else:
                    self.increments = [increment for _ in range(len(order) // increment)]
            elif ((self._opt.dataset == 'cifar100') and (self._opt.model=="iid2")):
                root_path = self._opt.data_path
                train_dataset = dataset.base_dataset(root_path, train=True, download=True)
                test_dataset = dataset.base_dataset(root_path, train=False, download=True)

                x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
                x_val, y_val, x_train, y_train = self._list_split_per_class(
                    x_train, y_train, validation_split
                )
                x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

                order = [i for i in range(len(np.unique(y_train)))]
                if class_order_type == 'random':
                    random.seed(seed)  # Ensure that following order is determined by seed:
                    random.shuffle(order)
                    print("Class order:", order)
                elif class_order_type == 'old' and dataset.class_order is not None:
                    order = dataset.class_order
                elif class_order_type == 'super' and dataset.class_order_super is not None:
                    order = dataset.class_order_super
                else:
                    print("Classes are presented in a chronological order")

                self.class_order.append(order)

                y_train = self._map_new_class_index(y_train, order)
                y_val = self._map_new_class_index(y_val, order)
                y_test = self._map_new_class_index(y_test, order)

                super_y_train = self._make_super_classes(y_train, self.increment)
                super_y_test = self._make_super_classes(y_test, self.increment)
                super_y_val = self._make_super_classes(y_val, self.increment)

                y_train += current_class_idx
                y_val += current_class_idx
                y_test += current_class_idx

                current_class_idx += len(order)
                if len(datasets) > 1:
                    self.increments.append(len(order))
                else:
                    self.increments = [increment for _ in range(len(order) // increment)]
        
            else:
                root_path = self._opt.data_path
                train_dataset = dataset.base_dataset_hierarchy(root_path, train=True, download=True)
                test_dataset = dataset.base_dataset_hierarchy(root_path, train=False, download=True)

                x_train, y_train, super_y_train = train_dataset.data, np.array(train_dataset.targets), np.array(train_dataset.super_targets)
                x_val, y_val, super_y_val, x_train, y_train, super_y_train = self._split_per_class(
                    x_train, y_train, super_y_train, validation_split
                )
                x_test, y_test, super_y_test = test_dataset.data, np.array(test_dataset.targets), np.array(test_dataset.super_targets)
                
                idxs = np.argsort(super_y_test)
                x_test = x_test[idxs]
                y_test = y_test[idxs]
                super_y_test = super_y_test[idxs]

                idxs = np.argsort(super_y_train)
                x_train = x_train[idxs]
                y_train = y_train[idxs]
                super_y_train = super_y_train[idxs]

                idxs = np.argsort(super_y_val)
                x_val = x_val[idxs]
                y_val = y_val[idxs]
                super_y_val = super_y_val[idxs]

                idxs = np.unique(y_test, return_index=True)[1]
                unique_y_order = [y_test[id] for id in sorted(idxs)]
                unique_supery_order = [super_y_test[id] for id in sorted(idxs)]
                print(unique_supery_order)
                print(unique_y_order)

                y_train = self._map_new_class_index(y_train, unique_y_order)
                y_val = self._map_new_class_index(y_val, unique_y_order)
                y_test = self._map_new_class_index(y_test, unique_y_order)

                
                y_train += current_class_idx
                y_val += current_class_idx
                y_test += current_class_idx

                # current_class_idx += len(order)
                # if len(datasets) > 1:
                #     raise(Exception("current_class_idx doesnt work for more than one dataset right now, correct it"))
                #     self.increments.append(len(order))
                # else:
                self.increments = [increment for _ in range(20)]

            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.super_targets_train.append(super_y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.super_targets_val.append(super_y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)
            self.super_targets_test.append(super_y_test)

            # print(self.increments)

            self.data_train = np.concatenate(self.data_train)
            self.targets_train = np.concatenate(self.targets_train)
            self.super_targets_train = np.concatenate(self.super_targets_train)
            self.data_val = np.concatenate(self.data_val)
            self.targets_val = np.concatenate(self.targets_val)
            self.super_targets_val = np.concatenate(self.super_targets_val)
            self.data_test = np.concatenate(self.data_test)
            self.targets_test = np.concatenate(self.targets_test)
            self.super_targets_test = np.concatenate(self.super_targets_test)

    def _make_super_classes(self, y, increment):
        unique_y = np.unique(y)
        super_y = [int(i/increment) for i in range(len(unique_y))]
        super_order = [super_y[y[i]] for i in range(len(y))]
        return super_order
    
    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))


    @staticmethod
    def _split_per_class(x, y, super_y, validation_split=0.):
        """Splits train data for a subset of validation data.
        Split is done so that each class has same amount of data.
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        # idxs = np.argsort(y)
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]
        super_y = super_y[shuffled_indexes]

        x_val, y_val, super_y_val = [], [], []
        x_train, y_train, super_y_train = [], [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            super_y_val.append(super_y[val_indexes])

            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])
            super_y_train.append(super_y[train_indexes])


        x_val, y_val, super_y_val = np.concatenate(x_val), np.concatenate(y_val), np.concatenate(super_y_val) 
        x_train, y_train, super_y_train = np.concatenate(x_train), np.concatenate(y_train), np.concatenate(super_y_train) 

        return x_val, y_val, super_y_val, x_train, y_train, super_y_train

    @staticmethod
    def _list_split_per_class(x, y, validation_split=0.):
        """Splits train data for a subset of validation data.
        Split is done so that each class has a much data.
        """
        c = list(zip(x, y))
        random.shuffle(c)
        x, y = zip(*c)

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val_i = [x[i] for i in val_indexes]
            y_val_i = [y[i] for i in val_indexes]
            
            x_train_i = [x[i] for i in train_indexes]
            y_train_i = [y[i] for i in train_indexes]
            
            x_val.append(x_val_i)
            y_val.append(y_val_i)
            
            x_train.append(x_train_i)
            y_train.append(y_train_i)
            
        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_val, y_val, x_train, y_train

###### Unused functions
    def get_idx_data(self, idx, batch_size, mode="test", data_source="train"):
        """Returns a custom loader with specific idxs only.
        :param idx: A list of data indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if data_source == "train":
            x, y, super_y = self.data_train, self.targets_train, self.super_targets_train
        elif data_source == "val":
            x, y, super_y = self.data_val, self.targets_val, self.super_targets_val
        elif data_source == "test":
            x, y, super_y = self.data_test, self.targets_test, self.super_targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))
        y, sorted_idx = y.sort()

        sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
        trsf = transforms.Compose(self.common_transforms)

        loader = DataLoader(
            DummyDataset(x[sorted_idx], y, trsf, super_y=super_y[sorted_idx]),
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._workers)

    def get_custom_loader(self, class_indexes, mode="test", data_source="train"):
        """Returns a custom loader.
        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y, super_y = self.data_train, self.targets_train, self.super_targets_train
        elif data_source == "val":
            x, y, super_y = self.data_val, self.targets_val, self.super_targets_val
        elif data_source == "test":
            x, y, super_y = self.data_test, self.targets_test, self.super_targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets, super_targets = [], [], []
        for class_index in class_indexes:
            class_data, class_targets, super_class_targets  = self._select(
                x, y, super_y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)
            super_targets.append(super_class_targets)            

        data = np.concatenate(data)
        targets = np.concatenate(targets)
        super_targets = np.concatenate(super_targets)

        return data, self._get_loader(data, targets, super_targets, shuffle=False, mode=mode)
    
