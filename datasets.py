"""
Abstraction for loading datasets: currently supports SVHN and
we plan to support MNIST and more.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import math
from PIL import Image


class SVHNVec(torchvision.datasets.svhn.SVHN):
    """
    Custom SVHN class to allow for probability vector targets instead of just integers.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]  # this is the change
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# todo: add fancy-formatted documentation
# note: how can we document input arguments if the user doesn't directly call this?
# solution: we document it in the helper/creation function, which will be defined in the end
class _Dataset:
    '''
    This is a class that abstracts the torch dataset class.
    TODO: finish documenting

    ...

    Attributes
    ----------
    name : string
        communicating in a readable way which dataset we are using
    num_teachers : int
        representing the number of teachers
    num_labels : int
        representing the number of labels that the datapoints can have
    input_shape : tuple
        representing the shape of a datapoint
    '''
    def __init__(self):
        """
        Initializer function for the _Dataset class.
        
        :param num_teachers: int representing the number of teachers that we are training
                             and then aggregating votes over
        :param seed: 
        """
        raise NotImplementedError

        self.teach_train = None
        """
        Returns array of datasets for the teacher's to train on:
        has length equal to number of teachers, and randomized
        splits based on the initializing generator. See `teach_valid`
        for how to use the validation accuracy as well.
        """

        self.teach_valid = None
        # note: the code block formatting isn't really working
        """
        Returns array of validation datasets for the teachers. To use this in
        combination with `teach_train` for training, do something like this:
        ```python
        teach_train = datasets.svhn.teach_train()
        teach_valid = datasets.svhn.teach_valid()
        for train_set, valid_set in zip(teach_train, teach_valid):
            # ... train teacher i
        ```
        """

        self.teach_test = None
        """
        Returns the test set for teachers. Not sure if we'll use this, but good to have anyways!
        """

        self.student_data = None
        """
        Returns the data which a student will use to improve. This lumps together
        the student training and validation data. DO NOT MODIFY THIS. (Please!)
        It's meant for the teachers to read from when aggregating: when you want
        to overwrite the labels, please use `student_overwrite_labels`.
        """

        self.student_test = None
        """
        Returns the test set for the student. This should not be trained or validated on ---
        it's just for a final accuracy report.
        """

    # future: can maybe pass "semi_supervise=True" and return both labeled and unlabeled data
    def student_overwrite_labels(self, labels, semi_supervise=False):
        """
        Given a set of labels which correspond by index to the data given by `student_data`,
        this copies the data, overwrites the labels, and returns *both the student training
        and validation data* as a tuple (appropriately partitioned 80/20 and removing unlabeled data).
        """
        raise NotImplementedError

    def one_hot(self, labels):
        """
        Turn a vector of 1-d labels into a vector of
        one-hots (output a 2d array).
        """
        # keeping old None-handling code for now
        # label_vecs = np.full((len(labels), self.num_labels), None)
        # eye = np.eye(self.num_labels)
        # label_vecs[labels != None] = eye[labels[labels != None].astype(int)]  # noqa: E711
        # return label_vecs
        if isinstance(labels, np.ndarray):
            eye = np.eye(self.num_labels)
        else:  # it's a tensor
            eye = torch.eye(self.num_labels, device=labels.device)
        return eye[labels]


# child classes will only have implementation code, not docstrings


# note: this loads everything all at once, we could do
# functools.cached_property stuff to make it nicer
# but that's not critical right now!
class _Svhn(_Dataset):
    '''
    This is a class that abstracts the torch dataset class.
    TODO: finish documenting

    ...

    Attributes
    ----------
    name : string
        communicating in a readable way which dataset we are using
    num_teachers : int
        representing the number of teachers
    num_labels : int
        representing the number of labels that the datapoints can have
    input_shape : tuple
        representing the shape of a datapoint
    teach_train : list of torch.Subset objects
        representing a partition of the dataset according to self.num_teachers
        used for training data
    teach_valid : list of torch.Subset objects
        representing a partition of the dataset according to self.num_teachers
        used for validation data
    teach_test : SVHNVec object
        used to test teacher accuracy
    student_data : dataset
        representing the data used to train the student model. note that this
        has the true labels, and thus cannot be given to the student without
        constituting a privacy violation
    student_test : dataset
        used to test the student model

    Methods
    ---------
    one_hot(labels):
        Function that takes in an array of labels and returns an array of
        one-hot vectors that represent the labels
    student_overwrite_labels(labels)
        Function that takes in an array of labels which correspond to
        datapoints in self.student_data, and then overwrites with the labels
        that we have (not necessarily labelling every datapoint) and returns
        the dataset partitioned into train and validation data

    '''

    def __init__(self, num_teachers, seed=None):
        self._generator = torch.Generator()
        if seed is not None:
            self._generator = self._generator.manual_seed(seed)

        self.name = "svhn"
        self.num_teachers = num_teachers
        self.num_labels = 10
        self.input_shape = (32, 32, 3)
        # note: we're just storing the input shape instead of all the layers, but
        # maybe in the future that would change
        # self.layers = []

        tfs = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
        self._transform = transforms.Compose(tfs)
        # note: we're still in debate about normalization
        self._transform_normalize = transforms.Compose(
            tfs
            + [
                transforms.Normalize(
                    [0.4376821, 0.4437697, 0.47280442],
                    [0.19803012, 0.20101562, 0.19703614],
                )
            ]
        )
        # we normalize the input to the teachers, but not the student
        og_train = SVHNVec(
            "./data/svhn",
            split="train",
            download=True,
            transform=self._transform_normalize,
        )
        og_extra = SVHNVec(
            "./data/svhn",
            split="extra",
            download=True,
            transform=self._transform_normalize,
        )
        og_test = SVHNVec(
            "./data/svhn", split="test", download=True, transform=self._transform
        )
        # make all labels one-hot
        og_train.labels = self.one_hot(og_train.labels)
        og_extra.labels = self.one_hot(og_extra.labels)
        og_test.labels = self.one_hot(og_test.labels)
        # first, randomly split the train+extra into train and valid collections
        all_teach_train, all_teach_valid = torch.utils.data.random_split(
            torch.utils.data.ConcatDataset([og_train, og_extra]),
            [0.8, 0.2],
            generator=self._generator,
        )

        train_size = len(all_teach_train)
        valid_size = len(all_teach_valid)

        # then partition them into num_teachers selections
        train_partition = [
            math.floor(train_size / num_teachers) + 1
            for i in range(train_size % num_teachers)
        ] + [
            math.floor(train_size / num_teachers)
            for i in range(num_teachers - (train_size % num_teachers))
        ]
        valid_partition = [
            math.floor(valid_size / num_teachers) + 1
            for i in range(valid_size % num_teachers)
        ] + [
            math.floor(valid_size / num_teachers)
            for i in range(num_teachers - (valid_size % num_teachers))
        ]

        # now assign self variables to those (these are what the "user" will access!)
        self.teach_train = torch.utils.data.random_split(
            all_teach_train, train_partition, generator=self._generator
        )
        self.teach_valid = torch.utils.data.random_split(
            all_teach_valid, valid_partition, generator=self._generator
        )
        self.teach_test = og_test

        # future: can we randomly split the student learning and test data?
        # would need to somehow keep track of the indices for teacher labeling
        student_data_len = math.floor(len(og_test) * 0.9)
        self.student_data = torch.utils.data.Subset(
            og_test, torch.arange(student_data_len)
        )
        self.student_test = torch.utils.data.Subset(
            og_test, torch.arange(student_data_len, len(og_test))
        )

    def student_overwrite_labels(self, labels, semi_supervise=False):
        # note: this is some duplicated code
        # we re-load so we don't modify labels of other references
        og_test = SVHNVec(
            "./data/svhn", split="test", download=True, transform=self._transform
        )
        student_data_len = math.floor(len(og_test) * 0.9)
        student_data = torch.utils.data.Subset(og_test, torch.arange(student_data_len))
        unlabeled_data = torch.utils.data.Subset(og_test, torch.arange(student_data_len))
        # end duplicated code
        # note: labels should be length of full test set
        assert len(labels) == len(student_data), 'input "labels" not the correct length'

        labeled_indices = np.any(labels != None, axis=1)  # noqa: E711
        unlabeled_indices = np.any(labels == None, axis=1) # noqa: E711
        student_data.indices = student_data.indices[labeled_indices]
        unlabeled_data.indices = unlabeled_data.indices[unlabeled_indices]
        # FIXME below isn't happy because of shape
        student_data.dataset.labels = np.eye(self.num_labels)[
            student_data.dataset.labels
        ]
        student_data.dataset.labels[student_data.indices] = labels[labeled_indices]
        unlabeled_data.dataset.labels[unlabeled_data.indices] = np.full((len(unlabeled_data.indices),self.num_labels), None)

        # check: does this work? I think so but not 100% sure
        stud_train, stud_valid = torch.utils.data.random_split(
            student_data, [0.8, 0.2], generator=self._generator
        )
        if semi_supervise:
            return stud_train, stud_valid, unlabeled_data
        return stud_train, stud_valid


def make_dataset(dataset_name, num_teachers, seed=None):
    """
    todo add documentation!
    """
    # match-case to only accept valid dataset strings
    match dataset_name:
        case "svhn":
            return _Svhn(num_teachers, seed)
        case "mnist":
            return None  # TODO: implement MNIST!
        case _:
            raise Exception(f'no support for making dataset "{dataset_name}"')
