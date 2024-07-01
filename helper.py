import torch
import torchvision
import torchvision.transforms.v2 as transforms

# set up device; print the first time we use it
# a bit cursed but it's fine
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
try:
    already_printed  # noqa: F821
except:  # noqa: E722
    already_printed = True
    print("using device", device)

def load_dataset(dataset_name = 'svhn', split='teach', make_normal=False):
    """
    Function for loading the datasets that we need to load.
    :param dataset_name: string specifying which dataset to load (currently, one of 'svhn' or 'mnist'). defaults to 'svhn'
    :param split: string specifying how to split up the data, depending on if we're training the student or the teacher models. defaults to 'teach'
    :param make_normal: boolean specifying whether or not to normalize the data. defaults to False
    :return: 3-tuple containing the training, validation, and test datasets.
    """
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
    if dataset_name == 'svhn':
        if split == 'teach':
            train_dataset = torchvision.datasets.SVHN('./data/svhn', split='train', download=True, transform=transform)
            extra_dataset = torchvision.datasets.SVHN('./data/svhn', split='extra', download=True, transform=transform)
            test_dataset = torchvision.datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset,extra_dataset])
            train_length = int(len(train_dataset) * 0.8)
            valid_length = len(train_dataset) - train_length
            train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(train_length))
            valid_dataset = torch.utils.data.Subset(train_dataset, torch.arange(train_length, valid_length + train_length))
        else:
            all_data = torchvision.datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
            train_length = int(len(all_data)*0.7)
            valid_length = int(len(all_data)*0.2)
            train_dataset = torch.utils.data.Subset(all_data, torch.arange(train_length))
            valid_dataset = torch.utils.data.Subset(all_data, torch.arange(train_length, train_length + valid_length))
            test_dataset = torch.utils.data.Subset(all_data, torch.arange(train_length + valid_length, len(all_data)))
        if make_normal:
            normalize = transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])
            train_dataset, valid_dataset, test_dataset = normalize(train_dataset), normalize(valid_dataset), normalize(test_dataset)
    elif dataset_name == 'mnist':
        if split == 'teach':
            train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
            train_length = int(len(train_dataset) * 0.8)
            valid_length = len(train_dataset) - train_length
            train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(train_length))
            valid_dataset = torch.utils.data.Subset(train_dataset, torch.arange(train_length, valid_length + train_length))
        else:
            all_data = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
            train_length = int(len(all_data)*0.7)
            valid_length = int(len(all_data)*0.2)
            train_dataset = torch.utils.data.Subset(all_data, torch.arange(train_length))
            valid_dataset = torch.utils.data.Subset(all_data, torch.arange(train_length, train_length + valid_length))
            test_dataset = torch.utils.data.Subset(all_data, torch.arange(train_length + valid_length, len(all_data)))
        if make_normal:
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            train_dataset, valid_dataset, test_dataset = normalize(train_dataset), normalize(valid_dataset), normalize(test_dataset)
    else:
        print("Bad value of dataset arg.")
        return False
    return train_dataset, valid_dataset, test_dataset