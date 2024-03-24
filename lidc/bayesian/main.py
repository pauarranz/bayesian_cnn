import pathlib

from dataset_torch import LidcNoduleDataset, ToTensorWithOriginalShape, rotate_image
from model import BayesianLidcNodulesNet

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchinfo import summary

import os

from pathlib import Path


def save_models(models, save_dir):
	for i, m in enumerate(models):
		file_name = os.path.join(save_dir, "model{}.pth".format(i))

		torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(file_name))


def load_models(save_dir):
	models = []

	for f in os.listdir(save_dir):
		model = BayesianLidcNodulesNet(p_mc_dropout=None)
		model.load_state_dict(torch.load(os.path.abspath(os.path.join(save_dir, f)))["model_state_dict"])
		models.append(model)

	return models


class RunExperiment:
	def __init__(
			self,
			filtered_class: int = 5,
			save_dir: pathlib.Path = None,
			no_train: bool = True,
			n_epochs: int = 10,
			n_batch: int = 64,
			n_runtests: int = 50,
			learning_rate=5e-3,
			num_networks: int = 10,
	):
		"""

		:param filtered_class: The class to load from data
		:param save_dir: Directory where the models can be saved or loaded from.
		:param no_train: Load the models directly instead of training.
		:param n_epochs: The number of epochs to train for.
		:param n_batch: Batch size used for training.
		:param n_runtests: The number of pass to use at test time for monte-carlo uncertainty estimation.
		:param learning_rate: The learning rate of the optimizer.
		:param num_networks: The number of networks to train to make an ensemble.
		"""
		self.filtered_class = filtered_class
		self.save_dir = save_dir
		self.no_train = no_train
		self.n_epochs = n_epochs
		self.n_batch = n_batch
		self.n_runtests = n_runtests
		self.learning_rate = learning_rate
		self.num_networks = num_networks

		self.models = []
		self.test_filtered = None
		self.N = None
		self.train_loader = None
		self.test_loader = None

	def load_data(self, data_dir: pathlib.Path):
		"""

		:param data_dir: Directory where the data is saved.
		:return:
		"""

		# Create dataset instance with augmentation
		dataset = LidcNoduleDataset(
			data_dir,
			num_slices=6,
			transform=ToTensorWithOriginalShape(),
			augment=True,
			filter_label=self.filtered_class,
		)
		print(f'Data shape: {np.array(dataset[0][0]).shape}')
		# Divide between train and test dataset
		train, test = torch.utils.data.random_split(dataset, lengths=[0.7, 0.3])

		# Create iterable from the dataset to apply to the model
		self.train_loader = torch.utils.data.DataLoader(train, batch_size=self.n_batch)
		self.test_loader = torch.utils.data.DataLoader(test, batch_size=self.n_batch)

		self.N = len(train)

	def get_models(self):
		if self.no_train:
			# Load models
			self.models = load_models(self.save_dir)
		else:
			# Train models
			batch_len = len(self.train_loader)
			digits_batch_len = len(str(batch_len))
			for i in np.arange(self.num_networks):
				print("Training model {}/{}:".format(i + 1, self.num_networks))

				# Initialize the model
				model = BayesianLidcNodulesNet(
					p_mc_dropout=None)  # p_mc_dropout=None will disable MC-Dropout for this bnn, as we found out it makes learning much much slower.
				summary(model, input_size=(self.n_batch, 1, 6, 64, 64))
				loss = torch.nn.NLLLoss(reduction='mean')  # negative log likelihood will be part of the ELBO

				optimizer = Adam(model.parameters(), lr=self.learning_rate)
				optimizer.zero_grad()

				for n in np.arange(self.n_epochs):

					for batch_id, sampl in enumerate(iter(self.train_loader)):
						images, labels = sampl
						#print(f"Feature batch shape: {images.size()}")
						#print(f"Labels batch shape: {labels.size()}")

						pred = model(images, stochastic=True)

						logprob = loss(pred, labels)
						l = self.N * logprob

						modelloss = model.evalAllLosses()
						l += modelloss

						optimizer.zero_grad()
						l.backward()

						optimizer.step()

						print(
							"\r",
							("\tEpoch {}/{}: Train step {" + (":0{}d".format(digits_batch_len)
							) + "}/{} prob = {:.4f} model = {:.4f} loss = {:.4f}          "
							).format(
								n + 1, self.n_epochs,
								batch_id + 1,
								batch_len,
								torch.exp(-logprob.detach().cpu()).item(),
								modelloss.detach().cpu().item(),
								l.detach().cpu().item()),
							end="",
						)
				print("")

				self.models.append(model)

			if self.save_dir is not None:
				save_models(self.models, self.save_dir)

	def test_models(self):
		with torch.no_grad():

			samples = torch.zeros((self.n_runtests, self.test_loader.batch_size, 2))

			images, labels = next(iter(self.test_loader))

			for i in np.arange(self.n_runtests):
				print("\r", "\tTest run {}/{}".format(i + 1, self.n_runtests), end="")
				model = np.random.randint(self.num_networks)
				model = self.models[model]

				pred = model(images)
				pred_exp = torch.exp(pred)
				samples[i, :, :] = pred_exp

			withinSampleMean = torch.mean(samples, dim=0)
			samplesMean = torch.mean(samples, dim=(0, 1))

			withinSampleStd = torch.sqrt(torch.mean(torch.var(samples, dim=0), dim=0))
			acrossSamplesStd = torch.std(withinSampleMean, dim=0)

			print("\n\nClass prediction analysis:")
			print("\tMean class probabilities:")
			print(samplesMean)
			print("\tPrediction standard deviation per sample:")
			print(withinSampleStd)
			print("\tPrediction standard deviation across samples:")
			print(acrossSamplesStd)


if __name__ == '__main__':
	exp = RunExperiment(
		no_train=True,
		save_dir=Path('C:\\Users\\pau_a\\Documents\\Python_scripts\\bayesian_convolutional_neural_network\\lidc\\bayesian\\models_w_augmentation'),
		filtered_class=1,
	)
	exp.load_data(data_dir=Path('F:\\master\\manifest-1600709154662\\nodules_16slices'))
	exp.get_models()
	exp.test_models()

	exp = RunExperiment(
		no_train=True,
		save_dir=Path(
			'C:\\Users\\pau_a\\Documents\\Python_scripts\\bayesian_convolutional_neural_network\\lidc\\bayesian\\models_w_augmentation'),
		filtered_class=0,
	)
	exp.load_data(data_dir=Path('F:\\master\\manifest-1600709154662\\nodules_16slices'))
	exp.get_models()
	exp.test_models()
