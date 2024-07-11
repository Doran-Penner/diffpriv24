"""
Abstraction for loading datasets: WIP, but MVP will support SVHN and MNIST
with potential for more.
"""

import torch


# todo: add fancy-formatted documentation
class _Dataset:
    def __init__(self):
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


class _Svhn(_Dataset):
    # todo: docstrings not showing up for vscode? (gotta test more to be sure)
    # possible solutions:
    # __doc__ = super().__doc__
    # doc argument to @property?
    # ... (idk)

    def __init__(self, num_teachers, num_labels, seed=None):
        pass  # todo: fill it out
