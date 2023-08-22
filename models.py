import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

# linear mapping that enforces same-sign weights
class SameSignLinear(nn.Module):
	def __init__(self, in_features, out_features, sign = 1, weight = None, bias = 0.0):
		super(SameSignLinear, self).__init__()
		if weight is None:
			self.weight = nn.Parameter(torch.empty(out_features, in_features))
			nn.init.kaiming_uniform_(self.weight)
		elif type(weight) in [float, int]:
			assert in_features == out_features
			assert in_features == 1
			self.weight = nn.Parameter(torch.Tensor([[float(weight)]]))
		else:
			assert type(weight) == torch.Tensor
			self.weight = nn.Parameter(weight)
		self.bias = nn.Parameter(torch.Tensor([bias]))
		self.sign = sign

	def forward(self, input):
		return F.linear(input, self.sign*self.weight.abs(), self.bias)


class CosineClassifier(nn.Module):
	def __init__(self, dim, classes):
		super(CosineClassifier, self).__init__()
		self.weight = nn.Parameter(torch.Tensor(classes, dim))
		nn.init.kaiming_uniform_(self.weight)

	def forward(self, input):
		return F.linear(F.normalize(input), F.normalize(self.weight))


class ContrastiveWrapper(nn.Module):
	def __init__(self, config):
		super(ContrastiveWrapper, self).__init__()
		self.feat_dim = config['feature_dim']
		self.cnn = timm.create_model(config['model'], pretrained = config['pretrained'], num_classes=self.feat_dim)
		
		self.config = config
		if config['feature_activation'] == 'sigmoid':
			self.head = nn.Sigmoid()
		elif config['feature_activation'] == 'tanh':
			self.head = nn.Tanh()
		else:
			self.head = nn.Identity()
		
		# extras
		if config['loss'] in [
								'distance_classifier_bce',
								'distance_classifier_ce',
								'margin_classifier',
								'triplet_classifier',
								]:
			assert config['feature_activation'] == 'none'
			try:
				_bn = config['classifier_bn']
			except KeyError:
				_bn = True
			if _bn:
				self.extras = nn.ModuleDict({'classifier' : nn.BatchNorm1d(1)})
			else:
				self.extras = nn.ModuleDict({'classifier' : SameSignLinear(1, 1, sign=1, weight=1.0)})
		elif 'thermostat' in config['loss']:
			self.extras = nn.ModuleDict({'classifier' : SameSignLinear(1, 1, sign=1, weight=config['temperature'])})
		elif config['loss'] in ['softmax']:
			self.extras = nn.ModuleDict({'classifier' : nn.Linear(self.feat_dim, config['num_training_patients'])})
		elif config['loss'] in ['cos_softmax']:
			self.extras = nn.ModuleDict({'classifier' : CosineClassifier(self.feat_dim, config['num_training_patients'])})
		else:
			self.extras = None
	
	def forward(self, batch):
		device = next(self.parameters()).device
		
		# total_images x D
		features = self.cnn(batch.to(device)).reshape(len(batch), self.feat_dim)
		features = self.head(features)
		
		return features