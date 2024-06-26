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
        print("Model",str(i))
        state_dict = torch.load('./saved/teacher_'+str(i)+'.txt',map_location=torch.device('cpu'))
        m = CNN()
        m.load_state_dict(state_dict)
        m.eval()

        ballot = [] # user i's voting record
        correct = 0
        guessed = 0

        for p, l in data:
            out = m(p)
            if torch.argmax(out) == l:
                correct += 1
            guessed += 1
            ballot.append(torch.argmax(out))
        votes.append(ballot)
        print(correct/guessed)
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
loader = torch.utils.data.DataLoader(public_dataset, shuffle=True, batch_size=1)
agg = aggregate.NoisyMaxAggregator(1)
labels = getPredictedLabels(loader,agg)

with open('./saved/teacher_predictions.txt','w') as f:
    for label in labels:
        f.write(f"{label}\n")
