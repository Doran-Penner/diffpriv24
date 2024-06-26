import numpy
import torch
from models import CNN
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
import aggregate

def getPredictedLabels(data, aggregator, num_models=250):
    votes = [] # final voting record
    for i in range(num_models):
        state_dict = torch.load('./saved/teacher_'+str(i)+'.txt')
        m = CNN()
        m.load_state_dict(state_dict)
        m.eval()

        ballot = [] # user i's voting record

        for p in data:
            ballot.append(m(p))
        votes.append(ballot)
    return aggregator.aggregate(torch.transpose(torch.Tensor(votes),0,1))

#setup device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = 'cpu'
print(device)

#setup transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

public_dataset = torchvision.datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
unlabeled_data = public_dataset[0][0]
agg = aggregate.NoisyMaxAggregator(1)
labels = getPredictedLabels(unlabeled_data,agg)
