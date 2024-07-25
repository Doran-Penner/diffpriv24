from torch import nn

class CNN(nn.Module):

    def __init__(self, dat_obj):
        super().__init__()
        pad = 'same'
        size1, size2, channels = dat_obj.input_shape
        layers = [nn.Conv2d(channels, 64, 5, padding=pad)]
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(3,stride=2,padding=1))
        layers.append(nn.LocalResponseNorm(4,alpha=0.001 / 9.0, beta=0.75))
        
        layers.append(nn.Conv2d(64, 128, 5, padding=pad))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.3))
        layers.append(nn.LocalResponseNorm(4,alpha=0.001 / 9.0, beta=0.75))
        layers.append(nn.MaxPool2d(3,stride=2,padding=1))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(size1*size2*8,384))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(384,192))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(192,dat_obj.num_labels))       

        self.layers = layers
        self.model = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.model(x)

class BayesCNN(nn.module):

    def __init__(self,dat_obj):
        
        # copying the same code as above but with a few differences outlined
        # in the bayesian CNNs paper(s)
        # https://github.com/Riashat/Deep-Bayesian-Active-Learning/blob/master/MC_Dropout_Keras/Dropout_Bald_Q10_N1000_Paper.py
        super().__init__()
        pad = 'same'
        size1, size2, channels = dat_obj.input_shape

        # Layers:
        # These are based on the image data active learning paper
        # even though the Bayesian CNN intro paper has different
        # Architecture. (this is based on the code for the paper
        # not the paper actual stuff)
        layers = [nn.Conv2d(channels,64,5,padding=pad)]
        layers.append(nn.ReLU())
        layers = [nn.Conv2d(64,128,5,padding=pad)]
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(3,stride=2,padding=1))
        layers.append(nn.Dropout(p=0.25))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(size1*size2*32,256)) # add optimizer for regularization?
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Linear(256,dat_obj.num_labels))
        layers.append(nn.Softmax())

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x): # taken from normal CNN bit
        return self.model(x)