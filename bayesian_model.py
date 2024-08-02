"""

new shit code Taken from https://github.com/kumar-shridhar/PyTorch-BayesianCNN [1]
old shit Code Taken from https://github.com/french-paragon/BayesianMnist/blob/master/viModel.py [2]

"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.distributions.normal import Normal
import Bayes_utils as utils

# from [1]/layers/misc.py

class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)



# from [1]/layers/BBB_LRT/BBBConv.py

class BBBConv2d(ModuleWrapper):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, priors=None):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1), # NOTE in Configs, they give -5 as posterior_rho_initial
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.conv2d(
            x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        act_var = 1e-16 + F.conv2d(
            x ** 2, self.W_sigma ** 2, bias_var, self.stride, self.padding, self.dilation, self.groups)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = utils.calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += utils.calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl



# from [1]/layers/BBB_LRT/BBBLinear.py

class BBBLinear(ModuleWrapper):

    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
                priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1), # NOTE in Configs, they give -5 as posterior_rho_initial
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = utils.calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += utils.calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

# from [1]/models/BayesianModels/Bayesian3Conv3FC.py

class BBB3Conv3FC(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, dat_obj, priors=None, layer_type='lrt', activation_type='softplus'):
        super(BBB3Conv3FC, self).__init__()



        self.num_classes = dat_obj.num_labels
        self.layer_type = layer_type
        self.priors = priors

#        if layer_type=='lrt':
#            BBBLinear = BBB_LRT_Linear
#            BBBConv2d = BBB_LRT_Conv2d
#        elif layer_type=='bbb':
#            BBBLinear = BBB_Linear
#            BBBConv2d = BBB_Conv2d
#        else:            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(3, 32, 5, padding=2, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, 5, padding=2, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(64, 128, 5, padding=1, bias=True, priors=self.priors)
        self.act3 = self.act()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = FlattenLayer(2 * 2 * 128)
        self.fc1 = BBBLinear(2 * 2 * 128, 1000, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc2 = BBBLinear(1000, 1000, bias=True, priors=self.priors)
        self.act5 = self.act()

        self.fc3 = BBBLinear(1000, 10, bias=True, priors=self.priors)


# from [1]/models/BayesianModels/BayesianAlexNet.py

class BBBAlexNet(ModuleWrapper):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, dat_obj, priors = None, layer_type='lrt', activation_type='softplus'):
        super(BBBAlexNet, self).__init__()

        self.num_classes = dat_obj.num_labels
        self.layer_type = layer_type
        self.priors = priors

#        if layer_type=='lrt':
#            BBBLinear = BBB_LRT_Linear
#            BBBConv2d = BBB_LRT_Conv2d
#        elif layer_type=='bbb':
#            BBBLinear = BBB_Linear
#            BBBConv2d = BBB_Conv2d
#        else:            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(3, 64, 11, stride=4, padding=5, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(64, 192, 5, padding=2, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(192, 384, 3, padding=1, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.conv4 = BBBConv2d(384, 256, 3, padding=1, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.conv5 = BBBConv2d(256, 128, 3, padding=1, bias=True, priors=self.priors)
        self.act5 = self.act()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(1 * 1 * 128)
        self.classifier = BBBLinear(1 * 1 * 128, self.num_classes, bias=True, priors=self.priors)


# from [1]/models/BayesianModels/BayesianLeNet

class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

#        if layer_type=='lrt':
#            BBBLinear = BBB_LRT_Linear
#            BBBConv2d = BBB_LRT_Conv2d
#        elif layer_type=='bbb':
#            BBBLinear = BBB_Linear
#            BBBConv2d = BBB_Conv2d
#        else:            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, padding=0, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinear(5 * 5 * 16, 120, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.fc2 = BBBLinear(120, 84, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc3 = BBBLinear(84, outputs, bias=True, priors=self.priors)





# Old Shit
class VIModule(nn.Module) :
	"""
	A mixin class to attach loss functions to layer. This is usefull when doing variational inference with deep learning.
	"""
	
	def __init__(self, *args, **kwargs) :
		super().__init__(*args, **kwargs)
		
		self._internalLosses = []
		self.lossScaleFactor = 1
		
	def addLoss(self, func) :
		self._internalLosses.append(func)
		
	def evalLosses(self) :
		t_loss = 0
		
		for l in self._internalLosses :
			t_loss = t_loss + l(self)
			
		return t_loss
	
	def evalAllLosses(self) :
		
		t_loss = self.evalLosses()*self.lossScaleFactor
		
		for m in self.children() :
			if isinstance(m, VIModule) :
				t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
				
		return t_loss


class MeanFieldGaussianFeedForward(VIModule) :
	"""
	A feed forward layer with a Gaussian prior distribution and a Gaussian variational posterior.
	"""
	
	def __init__(self, 
			  in_features, 
			  out_features, 
			  bias = True,  
			  groups=1, 
			  weightPriorMean = 0, 
			  weightPriorSigma = 1.,
			  biasPriorMean = 0, 
			  biasPriorSigma = 1.,
			  initMeanZero = False,
			  initBiasMeanZero = False,
			  initPriorSigmaScale = 0.01) :
		
		
		super(MeanFieldGaussianFeedForward, self).__init__()
		
		self.samples = {'weights' : None, 'bias' : None, 'wNoiseState' : None, 'bNoiseState' : None}
		
		self.in_features = in_features
		self.out_features = out_features
		self.has_bias = bias
		
		self.weights_mean = Parameter((0. if initMeanZero else 1.)*(torch.rand(out_features, int(in_features/groups))-0.5))
		self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale*weightPriorSigma*torch.ones(out_features, int(in_features/groups))))

		self.noiseSourceWeights = Normal(torch.zeros(out_features, int(in_features/groups)), 
								   torch.ones(out_features, int(in_features/groups)))
		
		self.addLoss(lambda s : 0.5*s.getSampledWeights().pow(2).sum()/weightPriorSigma**2)
		self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())
		
		if self.has_bias :
			self.bias_mean = Parameter((0. if initBiasMeanZero else 1.)*(torch.rand(out_features)-0.5))
			self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale*biasPriorSigma*torch.ones(out_features)))
			
			self.noiseSourceBias = Normal(torch.zeros(out_features), torch.ones(out_features))
			
			self.addLoss(lambda s : 0.5*s.getSampledBias().pow(2).sum()/biasPriorSigma**2)
			self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())
			
			
	def sampleTransform(self, stochastic=True) :
		self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
		self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma)*self.samples['wNoiseState'] if stochastic else 0)
		
		if self.has_bias :
			self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
			self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma)*self.samples['bNoiseState'] if stochastic else 0)
		
	def getSampledWeights(self) :
		return self.samples['weights']
	
	def getSampledBias(self) :
		return self.samples['bias']
	
	def forward(self, x, stochastic=True) :
		
		self.sampleTransform(stochastic=stochastic)
		
		return nn.functional.linear(x, self.samples['weights'], bias = self.samples['bias'] if self.has_bias else None)
	
	
class MeanFieldGaussian2DConvolution(VIModule) :
	"""
	A Bayesian module that fit a posterior gaussian distribution on a 2D convolution module with normal prior.
	"""
	
	def __init__(self,
			  in_channels, 
			  out_channels, 
			  kernel_size, 
			  stride=1, 
			  padding=0, 
			  dilation=1, 
			  groups=1, 
			  bias=True, 
			  padding_mode='zeros', 
			  wPriorSigma = 1., 
			  bPriorSigma = 1.,
			  initMeanZero = False,
			  initBiasMeanZero = False,
			  initPriorSigmaScale = 0.01) :
		
		super(MeanFieldGaussian2DConvolution, self).__init__()
		
		self.samples = {'weights' : None, 'bias' : None, 'wNoiseState' : None, 'bNoiseState' : None}
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.has_bias = bias
		self.padding_mode = padding_mode
		
		
		self.weights_mean = Parameter((0. if initMeanZero else 1.)*(torch.rand(out_channels, int(in_channels/groups), self.kernel_size[0], self.kernel_size[1])-0.5))
		self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale*wPriorSigma*torch.ones(out_channels, int(in_channels/groups), self.kernel_size[0], self.kernel_size[1])))
			
		self.noiseSourceWeights = Normal(torch.zeros(out_channels, int(in_channels/groups), self.kernel_size[0], self.kernel_size[1]), 
								   torch.ones(out_channels, int(in_channels/groups), self.kernel_size[0], self.kernel_size[1]))
		
		self.addLoss(lambda s : 0.5*s.getSampledWeights().pow(2).sum()/wPriorSigma**2)
		self.addLoss(lambda s : -self.out_channels/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())
		
		
		if self.has_bias :
			self.bias_mean = Parameter((0. if initBiasMeanZero else 1.)*(torch.rand(out_channels)-0.5))
			self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale*bPriorSigma*torch.ones(out_channels)))
			
			self.noiseSourceBias = Normal(torch.zeros(out_channels), torch.ones(out_channels))
			
			self.addLoss(lambda s : 0.5*s.getSampledBias().pow(2).sum()/bPriorSigma**2)
			self.addLoss(lambda s : -self.out_channels/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())
			
			
	def sampleTransform(self, stochastic=True) :
		self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
		self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma)*self.samples['wNoiseState'] if stochastic else 0)
		
		if self.has_bias :
			self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
			self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma)*self.samples['bNoiseState'] if stochastic else 0)

	def getSampledWeights(self) :
		return self.samples['weights']
	
	def getSampledBias(self) :
		return self.samples['bias']
	
	def forward(self, x, stochastic=True) :
		
		self.sampleTransform(stochastic=stochastic)
		
		if self.padding != 0 and self.padding != (0,0) :
			padkernel = (self.padding, self.padding, self.padding, self.padding) if isinstance(self.padding, int) else (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
			mx = nn.functional.pad(x, padkernel, mode=self.padding_mode, value=0)
		else :
			mx = x
		breakpoint()
		return nn.functional.conv2d(mx, self.samples['weights'], bias = self.samples['bias'] if self.has_bias else None,stride=self.stride, padding='valid', dilation=self.dilation, groups=self.groups)
		
class BayesianNet(VIModule):
	# Changed the name to BayesianNet
	def __init__(self, 
			  	 dat_obj,
				 convWPriorSigma = 1., 
				 convBPriorSigma = 5., 
				 linearWPriorSigma = 1., 
				 linearBPriorSigma = 5., 
				 p_mc_dropout = 0.5) :
		
		super().__init__()
		
		# added dat_obj as an argument to __init__ for consistency with models.CNN
		size1,size2,channels = dat_obj.input_shape
		
		# in place for number of features (with channels per feature)
		# basically for SVHN and other image-datasets this will be number of pixels
		feats = size1 + size2

		self.p_mc_dropout = p_mc_dropout
		
		# changed the inputs to be in line with dat_obj
		self.conv1 = MeanFieldGaussian2DConvolution(channels, feats, 
													wPriorSigma = convWPriorSigma, 
													bPriorSigma = convBPriorSigma, 
													kernel_size=5,
													initPriorSigmaScale=1e-7)
		self.conv2 = MeanFieldGaussian2DConvolution(feats, 2*feats, 
													wPriorSigma = convWPriorSigma, 
													bPriorSigma = convBPriorSigma, 
													kernel_size=5,
													initPriorSigmaScale=1e-7)
		
		self.linear1 = MeanFieldGaussianFeedForward(512, 128,
													weightPriorSigma = linearWPriorSigma, 
													biasPriorSigma = linearBPriorSigma,
													initPriorSigmaScale=1e-7)
		self.linear2 = MeanFieldGaussianFeedForward(128, 10,
													weightPriorSigma = linearWPriorSigma, 
													biasPriorSigma = linearBPriorSigma,
													initPriorSigmaScale=1e-7)

	def forward(self, x, stochastic=True):
		
		x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x, stochastic=stochastic), 2))
		x = self.conv2(x, stochastic=stochastic)
		
		if self.p_mc_dropout is not None :
			x = nn.functional.dropout2d(x, p = self.p_mc_dropout, training=stochastic) #MC-Dropout
		
		x = nn.functional.relu(nn.functional.max_pool2d(x, 2))
		
		x = x.view(-1, 512)
		
		x = nn.functional.relu(self.linear1(x, stochastic=stochastic))
		
		if self.p_mc_dropout is not None :
			x = nn.functional.dropout(x, p = self.p_mc_dropout, training=stochastic) #MC-Dropout
		
		x = self.linear2(x, stochastic=stochastic)
		return nn.functional.log_softmax(x, dim=-1)
	
