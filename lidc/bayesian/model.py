#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:05:55 2021

@author: laurent
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch.distributions.normal import Normal


class VIModule(nn.Module):
	"""
	A mixin class to attach loss functions to layer. This is useful when doing variational inference with deep learning.
	"""
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		self._internalLosses = []
		self.lossScaleFactor = 1
		
	def addLoss(self, func):
		self._internalLosses.append(func)
		
	def evalLosses(self):
		t_loss = 0
		
		for l in self._internalLosses:
			t_loss = t_loss + l(self)
			
		return t_loss
	
	def evalAllLosses(self):
		
		t_loss = self.evalLosses()*self.lossScaleFactor
		
		for m in self.children():
			if isinstance(m, VIModule):
				t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
				
		return t_loss


class MeanFieldGaussianFeedForward(VIModule):
	"""
	A feed forward layer with a Gaussian prior distribution and a Gaussian variational posterior.
	"""

	def __init__(
			self,
			in_features,
			out_features,
			bias=True,
			groups=1,
			weightPriorMean=0,
			weightPriorSigma=1.,
			biasPriorMean=0,
			biasPriorSigma=1.,
			initMeanZero=False,
			initBiasMeanZero=False,
			initPriorSigmaScale=0.01):
		"""

		:param in_features:
		:param out_features:
		:param bias:
		:param groups:
		:param weightPriorMean:
		:param weightPriorSigma:
		:param biasPriorMean:
		:param biasPriorSigma:
		:param initMeanZero:
		:param initBiasMeanZero:
		:param initPriorSigmaScale:
		"""
		super(MeanFieldGaussianFeedForward, self).__init__()

		self.samples = {'weights': None, 'bias': None, 'wNoiseState': None, 'bNoiseState': None}

		self.in_features = in_features
		self.out_features = out_features
		self.has_bias = bias

		self.weights_mean = Parameter((0. if initMeanZero else 1.)*(torch.rand(out_features, int(in_features/groups))-0.5))
		self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale*weightPriorSigma*torch.ones(out_features, int(in_features/groups))))

		self.noiseSourceWeights = Normal(torch.zeros(out_features, int(in_features/groups)),
								   torch.ones(out_features, int(in_features/groups)))

		self.addLoss(lambda s: 0.5*s.getSampledWeights().pow(2).sum()/weightPriorSigma**2)
		self.addLoss(lambda s: -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())

		if self.has_bias:
			self.bias_mean = Parameter((0. if initBiasMeanZero else 1.)*(torch.rand(out_features)-0.5))
			self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale*biasPriorSigma*torch.ones(out_features)))

			self.noiseSourceBias = Normal(torch.zeros(out_features), torch.ones(out_features))

			self.addLoss(lambda s: 0.5*s.getSampledBias().pow(2).sum()/biasPriorSigma**2)
			self.addLoss(lambda s: -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())


	def sampleTransform(self, stochastic=True) :
		self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
		self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma)*self.samples['wNoiseState'] if stochastic else 0)

		if self.has_bias :
			self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
			self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma)*self.samples['bNoiseState'] if stochastic else 0)

	def getSampledWeights(self):
		return self.samples['weights']

	def getSampledBias(self):
		return self.samples['bias']

	def forward(self, x, stochastic=True):

		self.sampleTransform(stochastic=stochastic)

		return nn.functional.linear(x, self.samples['weights'], bias = self.samples['bias'] if self.has_bias else None)


class MeanFieldGaussian3DConvolution(VIModule):
	"""
	A Bayesian module that fit a posterior gaussian distribution on a 3D convolution module with normal prior.
	"""

	def __init__(
			self,
			in_channels,
			out_channels,
			kernel_size,
			stride=1,
			padding=0,
			dilation=1,
			groups=1,
			bias=True,
			padding_mode='zeros',
			wPriorSigma=1.,
			bPriorSigma=1.,
			initMeanZero=False,
			initBiasMeanZero=False,
			initPriorSigmaScale=0.01,
	):

		super(MeanFieldGaussian3DConvolution, self).__init__()

		self.samples = {'weights': None, 'bias': None, 'wNoiseState': None, 'bNoiseState': None}

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.has_bias = bias
		self.padding_mode = padding_mode

		self.weights_mean = Parameter(
			(0. if initMeanZero else 1.) * (torch.rand(out_channels, in_channels // groups, *self.kernel_size) - 0.5)
		)
		self.lweights_sigma = Parameter(
			torch.log(
				initPriorSigmaScale * wPriorSigma * torch.ones(out_channels, in_channels // groups, *self.kernel_size)
			)
		)

		self.noiseSourceWeights = Normal(
			torch.zeros(out_channels, in_channels // groups, *self.kernel_size),
			torch.ones(out_channels, in_channels // groups, *self.kernel_size),
		)

		self.addLoss(lambda s: 0.5 * s.getSampledWeights().pow(2).sum() / wPriorSigma ** 2)
		self.addLoss(lambda s: -self.out_channels / 2 * np.log(2 * np.pi) - 0.5 * s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())

		if self.has_bias:
			self.bias_mean = Parameter((0. if initBiasMeanZero else 1.) * (torch.rand(out_channels) - 0.5))
			self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale * bPriorSigma * torch.ones(out_channels)))

			self.noiseSourceBias = Normal(torch.zeros(out_channels), torch.ones(out_channels))

			self.addLoss(lambda s: 0.5 * s.getSampledBias().pow(2).sum() / bPriorSigma ** 2)
			self.addLoss(lambda s: -self.out_channels / 2 * np.log(2 * np.pi) - 0.5 * s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())

	def sampleTransform(self, stochastic=True):
		self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
		self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma) * self.samples['wNoiseState'] if stochastic else 0)

		if self.has_bias:
			self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
			self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma) * self.samples['bNoiseState'] if stochastic else 0)

	def getSampledWeights(self):
		return self.samples['weights']

	def getSampledBias(self):
		return self.samples['bias']

	def forward(self, x, stochastic=True):

		self.sampleTransform(stochastic=stochastic)

		if self.padding != 0 and self.padding != (0, 0, 0):
			padkernel = (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding) if isinstance(
				self.padding, int) else (
				self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0])
			mx = nn.functional.pad(x, padkernel, mode=self.padding_mode, value=0)
		else:
			mx = x

		return nn.functional.conv3d(
			mx,
			self.samples['weights'],
			bias=self.samples['bias'] if self.has_bias else None,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			groups=self.groups,
		)


class BayesianLidcNodulesNet(VIModule):
	def __init__(
			self,
			convWPriorSigma=1.,
			convBPriorSigma=5.,
			linearWPriorSigma=1.,
			linearBPriorSigma=5.,
			mc_dropout=False
	):
		"""Initialize all layers from the architecture

		Args:
			convWPriorSigma (float, optional): _description_. Defaults to 1..
			convBPriorSigma (float, optional): _description_. Defaults to 5..
			linearWPriorSigma (float, optional): _description_. Defaults to 1..
			linearBPriorSigma (float, optional): _description_. Defaults to 5..
			p_mc_dropout (float, optional): _description_. Defaults to 0.5.
		"""
		
		super().__init__()
		
		self.mc_dropout = mc_dropout

		# 1st convolution layer
		self.conv1 = MeanFieldGaussian3DConvolution(
			in_channels=1,
			out_channels=6,
			wPriorSigma=convWPriorSigma,
			bPriorSigma=convBPriorSigma,
			kernel_size=(2, 5, 5),
			initPriorSigmaScale=1e-7)
		# 2nd convolution layer
		self.conv2 = MeanFieldGaussian3DConvolution(
			in_channels=6,
			out_channels=16,
			wPriorSigma=convWPriorSigma,
			bPriorSigma=convBPriorSigma,
			kernel_size=(1, 5, 5),
			initPriorSigmaScale=1e-7)
		# 1st dense layer
		self.linear1 = MeanFieldGaussianFeedForward(
			in_features=2704,
			out_features=120,
			weightPriorSigma=linearWPriorSigma,
			biasPriorSigma=linearBPriorSigma,
			initPriorSigmaScale=1e-7)
		# 2nd dense layer
		self.linear2 = MeanFieldGaussianFeedForward(
			in_features=120,
			out_features=84,  # The number of classes to predict
			weightPriorSigma=linearWPriorSigma,
			biasPriorSigma=linearBPriorSigma,
			initPriorSigmaScale=1e-7)
		# 3rd dense layer
		self.linear3 = MeanFieldGaussianFeedForward(
			in_features=84,
			out_features=2,  # The number of classes to predict
			weightPriorSigma=linearWPriorSigma,
			biasPriorSigma=linearBPriorSigma,
			initPriorSigmaScale=1e-7)

	def forward(self, x, stochastic=True):
		
		# 1st convolution
		x = self.conv1(x, stochastic=stochastic)
		# MC-Dropout
		if self.mc_dropout:
			x = nn.functional.dropout3d(x, p=0.2, training=stochastic)
  		# Max pooling 3D
		x = nn.functional.relu(nn.functional.max_pool3d(input=x, kernel_size=2, stride=2))

		# 2nd convolution
		x = self.conv2(x, stochastic=stochastic)
		# Max pooling 3D
		x = nn.functional.relu(nn.functional.max_pool3d(input=x, kernel_size=2, stride=2))
		# MC-Dropout
		if self.mc_dropout:
			x = nn.functional.dropout3d(x, p=0.5, training=stochastic)
		
		# Flatten output
		batch_size = x.shape[0] # Get batch size
		x = x.view(batch_size, -1) # Reshape data to have one array for each image

		# 1st dense layer
		x = nn.functional.relu(self.linear1(x, stochastic=stochastic))
		
		# 2nd dense layer
		x = nn.functional.relu(self.linear2(x, stochastic=stochastic))

		# 3rd dense layer
		x = self.linear3(x, stochastic=stochastic)
		# Softmax
		x = nn.functional.log_softmax(x, dim=-1)

		return x
