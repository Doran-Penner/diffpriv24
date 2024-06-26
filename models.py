from torch import nn

class CNN(nn.Module):

    def __init__(self,padding=True,dataset='svhn'):
        super().__init__()
        pad = 'same' if padding else 0
        if dataset == 'svhn':
            size = 32
            layers = [nn.Conv2d(3,64,5, padding=pad)]
        elif dataset == 'mnist':
            size = 28
            layers = [nn.Conv2d(1,64,5, padding=pad)]
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(3,stride=2,padding=1))
        layers.append(nn.LocalResponseNorm(4,alpha=0.001 / 9.0, beta=0.75))
        
        layers.append(nn.Conv2d(64, 128, 5, padding=pad))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.3))
        layers.append(nn.LocalResponseNorm(4,alpha=0.001 / 9.0, beta=0.75))
        layers.append(nn.MaxPool2d(3,stride=2,padding=1))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(size*size*8,384))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(384,192))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(192,10))       

        self.layers = layers
        self.model = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.model(x)
