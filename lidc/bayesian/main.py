from dataset_torch import LidcNoduleDataset, ToTensorWithOriginalShape, rotate_image
from model import BayesianLidcNodulesNet

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
import pathlib
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as nnf

from torchinfo import summary


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
			filtered_class: int = None,
			save_dir: pathlib.Path = None,
			train: bool = True,
			n_epochs: int = 10,
			n_batch: int = 64,
			n_runtests: int = 50,
			learning_rate=5e-3,
			num_networks: int = 10,
	):
		"""

		:param filtered_class: The class to load from data
		:param save_dir: Directory where the models can be saved or loaded from.
		:param train: Load the models directly instead of training.
		:param n_epochs: The number of epochs to train for.
		:param n_batch: Batch size used for training.
		:param n_runtests: The number of pass to use at test time for monte-carlo uncertainty estimation.
		:param learning_rate: The learning rate of the optimizer.
		:param num_networks: The number of networks to train to make an ensemble.
		"""
		self.filtered_class = filtered_class
		self.save_dir = save_dir
		self.train = train
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

		# Check for GPU availability
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print("Device in use:", self.device)

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
		if not self.train:
			# Load models
			self.models = load_models(self.save_dir)

		else:

			# Train models
			batch_len = len(self.train_loader)
			digits_batch_len = len(str(batch_len))
			for i in np.arange(self.num_networks):
				print("Training model {}/{}:".format(i, self.num_networks))

				# Initialize the model
				model = BayesianLidcNodulesNet(
					p_mc_dropout=None, # p_mc_dropout=None will disable MC-Dropout for this bnn, as we found out it makes learning much much slower.
				).to(self.device)
				print(f'Model {i} device: ', self.device)
				summary(model, input_size=(self.n_batch, 1, 6, 64, 64))
				loss = torch.nn.NLLLoss(reduction='mean')  # negative log likelihood will be part of the ELBO

				optimizer = Adam(model.parameters(), lr=self.learning_rate)
				optimizer.zero_grad()

				for n in np.arange(self.n_epochs):

					for batch_id, sampl in enumerate(iter(self.train_loader)):
						images, labels = sampl
						images = images.to(self.device)
						labels = labels.to(self.device)

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

	def test_model_ensemble(self, output_dir):
		print('Model ensemble - evaluation started')
		# Initialize view class for visualizations generation
		vw = View(output_dir)

		# Initialize a variable to keep track of the model correct predictions
		model_correct = 0
		# Iterate over the test dataset
		nodule_id = 0
		pred_prob_dict = {'correct': [], 'incorrect': []}
		for images, labels in self.test_loader:
			images = images.to(self.device)
			labels = labels.to(self.device)
			nodule_id += 1
			pred = 0
			# Iterate over the models
			for model in self.models:
				# Perform inference with the current model
				with torch.no_grad():
					pred += model(images)
			avg_pred = pred / len(self.models)
			# Get the index of the maximum predicted probability for each sample
			pred_labels = torch.argmax(avg_pred, dim=1)
			# Count how many predictions match the true labels
			pred_correct = torch.sum(pred_labels == labels).item()
			model_correct += pred_correct

			# Compute prediction probability for all classes
			prob = nnf.softmax(avg_pred, dim=1)
			# Get probability for predicted class
			top_p, _ = prob.topk(1, dim=1)
			# Round up and convert to integer
			top_p = int(round(float(top_p[0][0]) * 100, 0))

			# Compute average probability for correct and incorrect predictions
			if bool(pred_correct):
				pred_prob_dict['correct'].append(top_p)
			else:
				pred_prob_dict['incorrect'].append(top_p)

		# Calculate the accuracy as the total correct predictions divided by the total number of samples
		accuracy = model_correct / len(self.test_loader.dataset)

		print(f"Model ensemble, accuracy: {accuracy * 100:.2f}%")

		# Plot prediction probability distribution
		vw.plot_pred_dist(
			filename=f'pred_prob_dist_model_ensemble.png',
			pred_prob_dict=pred_prob_dict,
			model_accuracy=accuracy * 100,
		)

		"""
		vw = View(output_dir)
		with torch.no_grad():

			samples = torch.zeros((self.n_runtests, self.test_loader.batch_size, 2))

			images, labels = next(iter(self.test_loader))
			images = images.to(self.device)
			labels = labels.to(self.device)

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
		"""

	def test_models_individually(self, output_dir, model_id=None):
		# Initialize view class for visualizations generation
		vw = View(output_dir)
		# Make that the model(s) computes and returns the gradcam
		for id, model in enumerate(self.models):
			if model_id is None or model_id == id:
				self.grad_cam_3d(model_id=id)

		# Iterate over the models
		for id, model in enumerate(self.models):
			if model_id is None or model_id == id:
				# Initialize a variable to keep track of the model correct predictions
				model_correct = 0
				# Iterate over the test dataset
				nodule_id = 0
				pred_prob_dict = {'correct': [], 'incorrect': []}
				for images, labels in self.test_loader:
					images = images.to(self.device)
					labels = labels.to(self.device)
					nodule_id += 1
					# Perform inference with the current model
					with torch.no_grad():
						pred, heatmap = model(images)
						# Get the index of the maximum predicted probability for each sample
						pred_labels = torch.argmax(pred, dim=1)
						# Count how many predictions match the true labels
						pred_correct = torch.sum(pred_labels == labels).item()
						model_correct += pred_correct

						# Compute prediction probability
						prob = nnf.softmax(pred, dim=1)
						top_p, _ = prob.topk(1, dim=1)
						top_p = int(round(float(top_p[0][0]) * 100, 0))

						# Prediction Grad-CAM
						vw.display_gradcam(
							filename=f'nodule-{nodule_id}_pred-{int(pred_labels)}_prob-{top_p}_correct-{bool(pred_correct)}.png',
							img=images[0][0],
							heatmap=heatmap[0][0],
							alpha=0.3,
							model_id=id,
						)

						# Compute average probability for correct and incorrect predictions
						if bool(pred_correct):
							pred_prob_dict['correct'].append(top_p)
						else:
							pred_prob_dict['incorrect'].append(top_p)

				# Calculate the accuracy as the total correct predictions divided by the total number of samples
				accuracy = model_correct / len(self.test_loader.dataset)

				print(f"Model {id}, accuracy: {accuracy * 100:.2f}%")

				# Plot prediction probability distribution
				vw.plot_pred_dist(
					filename=f'pred_prob_dist_model_{id}.png',
					pred_prob_dict=pred_prob_dict,
					model_accuracy=accuracy*100,
				)

	def grad_cam_3d(self, model_id):
		from medcam import medcam

		# Inject model with M3d-CAM
		self.models[model_id] = medcam.inject(
			self.models[model_id],
			output_dir="attention_maps",
			data_shape=[6, 64, 64],
			label='best',
			return_attention=True,
			save_maps=False,
		)


class View:
	def __init__(self, output_dir):
		self.output_dir = output_dir

	def display_gradcam(self, filename, img, heatmap, model_id, alpha=0.4):
		n_rows = 3
		n_images = 6
		fig, ax = plt.subplots(n_rows, 2, figsize=(10, 20))
		for image_num in range(0, n_images, 2):
			row_num = image_num // 2
			# Left image of row N
			img0 = ax[row_num, 0].imshow(np.squeeze(img[image_num]), cmap='gray')
			img1 = ax[row_num, 0].imshow(np.squeeze(heatmap[image_num]), cmap='jet', alpha=alpha, extent=img0.get_extent())
			# Right image of row N
			img0 = ax[row_num, 1].imshow(np.squeeze(img[image_num + 1]), cmap='gray')
			img1 = ax[row_num, 1].imshow(np.squeeze(heatmap[image_num + 1]), cmap='jet', alpha=alpha, extent=img0.get_extent())

		# Remove axis (visualize image only)
		plt.axis('off')

		# Make directory if not existing
		gradcam_dict = self.output_dir / f'gradcams_model_{model_id}'
		gradcam_dict.mkdir(parents=True, exist_ok=True)
		# Save figure
		plt.savefig(gradcam_dict / filename, bbox_inches='tight')
		plt.close()

	def plot_pred_dist(self, filename, pred_prob_dict, model_accuracy=None):
		plt.figure(figsize=(40, 15))
		# Plot probability histogram
		bins = np.linspace(50, 100, 50)

		# Correct predictions probability distribution
		plt.hist(
			pred_prob_dict['correct'],
			bins,
			weights=np.ones(len(pred_prob_dict['correct'])) / len(pred_prob_dict['correct']),
			alpha=0.5,
			label='correct',
			color='g',
		)
		# Incorrect predictions probability distribution
		plt.hist(
			pred_prob_dict['incorrect'],
			bins,
			weights=np.ones(len(pred_prob_dict['incorrect'])) / len(pred_prob_dict['incorrect']),
			alpha=0.5,
			label='incorrect',
			color='r',
		)
		# Add correct predictions mean probability
		plt.axvline(x=np.mean(pred_prob_dict['correct']), color='g', label='Correct predictions - mean probability')
		# Add incorrect predictions mean probability
		plt.axvline(x=np.mean(pred_prob_dict['incorrect']), color='r', label='Incorrect predictions - mean probability')
		if model_accuracy is not None:
			plt.axvline(x=model_accuracy, color='b', label='Model accuracy')
		# Format y axis as a percentage (example: from 0.8 to 80%)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
		# Add legend
		plt.legend(loc='upper right')
		plt.title('Prediction probability distribution')
		plt.xlabel('Prediction probability (%)')
		plt.ylabel('Distribution')
		plt.savefig(self.output_dir / filename, bbox_inches='tight')


if __name__ == '__main__':
	data_dir = Path('F:\\master\\manifest-1600709154662\\nodules_16slices')
	output_dir = Path('F:\\master\\manifest-1600709154662\\nodules_6_slices_gradcam_bayesian')

	exp = RunExperiment(
		train=True,
		n_batch=1,
		save_dir=output_dir / 'models_w_augmentation',
	)
	exp.load_data(data_dir)
	exp.get_models()

	exp.test_model_ensemble(output_dir=output_dir)
	exp.test_models_individually(
		output_dir=output_dir,
		model_id=None,
	)
