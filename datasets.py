"""
Abstraction for loading datasets: WIP, but MVP will support SVHN and MNIST
with potential for more.
"""


# todo: add documentation for each method here,
# so child classes will just have implementation code
# note: should we do fancy formatting? I think not for now, but later
class _Dataset:
    def __init__(self):
        # potential params: optional randomizer, num_labels, num_teachers
        # (since those are dataset_agnostic)
        pass
        # here we should load all the datasets on initialization,
        # and do any other stuff (todo)
    

    # note: fixed, small variables (like num labels) should be "flat" attributes,
    # as well as all non-modifying info (like teach_train)
    # methods should only exist for stuff that needs copying, i.e. label overwriting
    # however! we do @property stuff so we can document properly (hehe)
    # note: @property disallows modification, which isn't a problem right now
    # but I'm not a massive fan of removing user control
    
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
        """
        Returns array of validation datasets for the teachers. To use this
        in combination with `teach_train` for training, do something like this:
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
        Returns the test set for teachers
        """
        raise NotImplementedError
