"""
Abstraction for loading datasets: WIP, but MVP will support SVHN and MNIST
with potential for more.
"""

import torch
import torchvision


# todo: add fancy-formatted documentation
# note: how can we document input arguments if the user doesn't directly call this?
# solution: we document it in the helper/creation function, which will be defined in the end
class _Dataset:
    def __init__(self):
        """
        todo: document this. see _Svhn for args that we take
        """
        raise NotImplementedError
        # note: here's a basic outline for how to do stuff,
        # probably remove this later

        # generator = torch.Generator()
        # if seed is not None:
        #     generator = generator.manual_seed(seed)
        # self.num_teachers = num_teachers
        # self.num_labels = num_labels
        # self._teach_train = ... compute stuff
        # (we use the underscore so the user only
        # accesses the well-documented attribute)
        # self._layers = todo: how are we gonna do this and how abstract does it need to be?
        # self._transform = ...

    # we do @property stuff so we can document properly (hehe)
    # it essentially works the same as "flat" properties, so you can do
    # `x.teach_train` without parentheses to get the value
    @property
    def teach_train(self):
        """
        Returns array of datasets for the teacher's to train on:
        has length equal to number of teachers, and randomized
        splits based on the initializing generator. See `teach_valid`
        for how to use the validation accuracy as well.
        """
        raise NotImplementedError

    @property
    def teach_valid(self):
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
        raise NotImplementedError

    @property
    def teach_test(self):
        """
        Returns the test set for teachers. Not sure if we'll use this, but good to have anyways!
        """
        raise NotImplementedError

    @property
    def student_data(self):
        """
        Returns the data which a student will use to improve. This lumps together
        the student training and validation data. DO NOT MODIFY THIS. (Please!)
        It's meant for the teachers to read from when aggregating: when you want
        to overwrite the labels, please use `student_overwrite_labels`.
        """
        raise NotImplementedError

    @property
    def student_test(self):
        """
        Returns the test set for the student. This should not be trained or validated on ---
        it's just for a final accuracy report.
        """
        raise NotImplementedError

    # future: can maybe pass "semi_supervise=True" and return both labeled and unlabeled data
    def student_overwrite_labels(self, labels):
        """
        Given a set of labels which correspond by index to the data given by `student_data`,
        this copies the data, overwrites the labels, and returns *both the student training
        and validation data* as a tuple (appropriately partitioned 80/20 and removing unlabeled data).
        """
        raise NotImplementedError


# child classes will only have implementation code, not docstrings


# note: this loads everything all at once, we could do
# functools.cached_property stuff to make it nicer
# but that's not critical right now!
class _Svhn(_Dataset):
    # todo: docstrings not showing up for vscode? (gotta test more to be sure)
    # possible solutions:
    # __doc__ = super().__doc__
    # doc argument to @property?
    # ... (idk)

    def __init__(self, num_teachers, seed=None):
        self._generator = torch.Generator()
        if seed is not None:
            self._generator = self._generator.manual_seed(seed)

        self.name = "svhn"
        self.num_teachers = num_teachers
        self.num_labels = 10
        self.input_shape = (32, 32, 3)

        tfs = [
            torchvision.transforms.v2.ToImage(),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
        ]
        self._transform = torchvision.transforms.v2.Compose(tfs)
        transform_normalize = torchvision.transforms.v2.Compose(
            tfs
            + [
                torchvision.transforms.v2.Normalize(
                    [0.4376821, 0.4437697, 0.47280442],
                    [0.19803012, 0.20101562, 0.19703614],
                )
            ]
        )
        # we normalize the input to the teachers, but not the student
        og_train = torchvision.datasets.SVHN(
            "./data/svhn", split="train", download=True, transform=transform_normalize
        )
        og_extra = torchvision.datasets.SVHN(
            "./data/svhn", split="extra", download=True, transform=transform_normalize
        )
        og_test = torchvision.datasets.SVHN(
            "./data/svhn", split="test", download=True, transform=self._transform
        )
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
            torch.floor(train_size / num_teachers) + 1
            for i in range(train_size % num_teachers)
        ] + [
            torch.floor(train_size / num_teachers)
            for i in range(num_teachers - (train_size % num_teachers))
        ]
        valid_partition = [
            torch.floor(valid_size / num_teachers) + 1
            for i in range(valid_size % num_teachers)
        ] + [
            torch.floor(valid_size / num_teachers)
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
        student_data_len = torch.floor(len(og_test) * 0.9)
        self.student_data = torch.utils.data.Subset(
            og_test, torch.arange(student_data_len)
        )
        self.student_test = torch.utils.data.Subset(
            og_test, torch.arange(student_data_len, len(og_test))
        )

        # todo: work with Carter to figure out how best to handle this
        self.layers = []

        def student_overwrite_labels(self, labels):
            # feels like we're hard-coding this -1 encoding which may not be as good for other datasets
            # (e.g. regression) and also not as nice for randomly giving stuff to teachers to be labeled
            # however! I think that's ambitious to change, so we're not going to worry about that for now

            # note: this is some duplicated code
            # we re-load so we don't modify labels of other references
            og_test = torchvision.datasets.SVHN(
                "./data/svhn", split="test", download=True, transform=self._transform
            )
            student_data_len = torch.floor(len(og_test) * 0.9)
            student_data = torch.utils.data.Subset(
                og_test, torch.arange(student_data_len)
            )
            # end duplicated code
            # note: labels should be length of full test set
            assert len(labels) == len(
                student_data
            ), 'input "labels" not the correct length'

            labeled_indices = labels != -1  # array of bools
            student_data.indices = student_data.indices[labeled_indices]
            student_data.dataset.labels[student_data.indices] = labels[labeled_indices]

            # check: does this work? I think so but not 100% sure
            stud_train, stud_valid = torch.utils.data.random_split(
                student_data, [0.8, 0.2], generator=self._generator
            )
            return stud_train, stud_valid


def make_dataset(dataset_name, num_teachers, seed=None):
    # match-case to only accept valid dataset strings
    match dataset_name:
        case "svhn":
            return _Svhn(num_teachers, seed)
        case "mnist":
            return None  # TODO: implement MNIST!
        case _:
            raise Exception(f'no support for making dataset "{dataset_name}"')


# note: can we avoid this hard-coded 250 teachers?
# solution 1: "global" variables in globals.py
# solution 2: helper.py should call make_dataset (so we can consolidate vars)
svhn = make_dataset("svhn", 250)
