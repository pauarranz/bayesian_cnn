from dataset_torch import LidcNoduleDataset, ToTensorWithOriginalShape, rotate_image
from model import BayesianLidcNodulesNet

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
import pathlib
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as nnf

from torchinfo import summary


def save_models(models, save_dir):
	for i, m in enumerate(models):
		file_name = os.path.join(save_dir, "model{}.pth".format(i))

		torch.save({"model_state_dict": m.state_dict()}, os.path.abspath(file_name))


def load_models(save_dir, device):
	models = []

	for f in os.listdir(save_dir):
		model = BayesianLidcNodulesNet(p_mc_dropout=None)
		model.load_state_dict(torch.load(os.path.abspath(os.path.join(save_dir, f)))["model_state_dict"])
		models.append(model.to(device))

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

	def get_models(self, dropout=None):
		if not self.train:
			# Load models
			self.models = load_models(self.save_dir, self.device)

		else:

			# Train models
			batch_len = len(self.train_loader)
			digits_batch_len = len(str(batch_len))
			for i in np.arange(self.num_networks):
				print("Training model {}/{}:".format(i, self.num_networks))

				# Initialize the model
				model = BayesianLidcNodulesNet(
					mc_dropout=dropout, # mc_dropout=False will disable MC-Dropout for this bnn, as we found out it makes learning much much slower.
				).to(self.device)
				print(f'Model {i} device: ', self.device)
				# Provide model summary
				summary(model, input_size=(self.n_batch, 1, 6, 64, 64))

				# Loss
				loss = torch.nn.NLLLoss(reduction='mean')  # negative log likelihood will be part of the ELBO
				# Optimizer
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
		pred_mean_dict = {'correct': [], 'incorrect': []}
		pred_prob_mean_dict = {'correct': [], 'incorrect': []}
		pred_label_std_dict = {'correct': [], 'incorrect': []}
		pred_prob_std_dict = {'correct': [], 'incorrect': []}
		for images, labels in tqdm(self.test_loader):
			images = images.to(self.device)
			labels = labels.to(self.device)
			nodule_id += 1
			pred = 0
			# Iterate over the models
			pred_prob_list = []
			pred_label_list = []
			for model in self.models:
				# Perform inference with the current model
				with torch.no_grad():
					model_pred = model(images)

					# Apply softmax to create range 0-100% probability for each class
					pred_mean_prob = nnf.softmax(model_pred, dim=1)
					# Get predicted label
					_, top_c = pred_mean_prob.topk(1, dim=1)
					# Save values in a dictionary
					pred_prob_list.append(pred_mean_prob)
					pred_label_list.append(top_c)
			
			# Stack the predictions a single tensor object with all the predictions (instead of a list of tensors)
			pred_prob_tensor = torch.stack(pred_prob_list)
			pred_label_tensor = torch.stack(pred_label_list).float()

			# Compute the mean prediction probability across all models
			pred_prob_tensor_mean = torch.mean(pred_prob_tensor, dim=0) # Get mean probability for each class
			pred_prob, pred_label = pred_prob_tensor_mean.topk(1, dim=1) # Get the highest probability
			pred_prob_list = [int(round(float(i[0].item()) * 100, 0)) for i in pred_prob] # Round it up and convert to integer

			#########
			# Prediction probability variance
   			#########
	  		# Compute the variance
			pred_prob_std = torch.std(pred_prob_tensor*100, dim=0)
			# Get the varaince of the predicted class only
			pred_prob_std = torch.mean(pred_prob_std, dim=1)

			#########
			# Prediction label variance
   			#########
			# Compute the variance
			pred_label_std = torch.std(pred_label_tensor, dim=0)
			
			# Count how many predictions match the true labels
			pred_label = torch.stack([i[0] for i in pred_label]) # convert list of tensors to tensor
			pred_correct_list = (pred_label == labels)
			pred_correct_sum = torch.sum(pred_correct_list).item() # Count number of correct predictions
			model_correct += pred_correct_sum

			# Compute average probability for correct and incorrect predictions
			pred_correct_list = pred_correct_list.tolist()
			pred_label_list = pred_label.tolist()
			pred_label_std_list = [i[0] for i in pred_label_std.tolist()]
			pred_prob_std_list = pred_prob_std.tolist()
			for pred_correct, pred_label, pred_prob, pred_label_std, pred_prob_std in zip(pred_correct_list, pred_label_list, pred_prob_list, pred_label_std_list, pred_prob_std_list):
				if bool(pred_correct):
					pred_mean_dict['correct'].append(pred_label)
					pred_prob_mean_dict['correct'].append(pred_prob)
					pred_label_std_dict['correct'].append(pred_label_std)
					pred_prob_std_dict['correct'].append(pred_prob_std)
				
				else:
					pred_mean_dict['incorrect'].append(pred_label)
					pred_prob_mean_dict['incorrect'].append(pred_prob)
					pred_label_std_dict['incorrect'].append(pred_label_std)
					pred_prob_std_dict['incorrect'].append(pred_prob_std)

		# Calculate the accuracy as the total correct predictions divided by the total number of samples
		accuracy = model_correct / len(self.test_loader.dataset)

		print(f"Model ensemble, accuracy: {accuracy * 100:.2f}%")

		# Plot prediction probability distribution
		vw.plot_pred_dist(
			filename=f'pred_prob_dist_model_ensemble.png',
			pred_prob_dict=pred_prob_mean_dict,
		)

		vw.plot_pred_dist(
			filename=f'pred_prob_std_model_ensemble.png',
			pred_prob_dict=pred_prob_std_dict,
			bins=np.linspace(0, 50, 10),
			model_acc=False,
		)

		# Plot prediction probability distribution
		vw.plot_pred_dist(
			filename=f'pred_label_std_model_ensemble.png',
			pred_prob_dict=pred_label_std_dict,
			bins=np.linspace(0, 1, 10),
			model_acc=False,
		)
		vw.plot_scatter(
			filename=f'pred_prob_std_vs_mean_model_ensemble.png', 
			dict_mean=pred_prob_mean_dict, 
			dict_std=pred_prob_std_dict,
		)

		print(f'Mean correct prediction probability: {np.mean(pred_prob_mean_dict["correct"])} %')
		print(f'Mean incorrect prediction probability: {np.mean(pred_prob_mean_dict["incorrect"])} %')

		print(f'Mean correct prediction probability std dev: {np.mean(pred_prob_std_dict["correct"])} %')
		print(f'Mean incorrect prediction probability std dev: {np.mean(pred_prob_std_dict["incorrect"])} %')

		print(f'Mean correct prediction label std dev: {np.mean(pred_label_std_dict["correct"])} %')
		print(f'Mean incorrect prediction label std dev: {np.mean(pred_label_std_dict["incorrect"])} %')

		print('Done')


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
				for images, labels in tqdm(self.test_loader):
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

	def plot_pred_dist(self, filename, pred_prob_dict, bins = np.linspace(50, 100, 10), model_acc=True):
		"""Plot prediction metrics distribution on a histogram

		Args:
			filename (pathlib.Path): filename of the graph image to be generated.
			pred_prob_dict (dict): {'correct':[], 'incorrect':[]}
			bins (numpy.linspace, optional): define histogram plot bins. Defaults to np.linspace(50, 100, 10).
			model_acc (bool, optional): True if model accuracy to be added to the plot. Defaults to True.
		"""
		plt.figure(figsize=(15, 5))
		# Plot probability histogram

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
		mean_prob_co = np.mean(pred_prob_dict['correct'])
		plt.axvline(x=np.mean(pred_prob_dict['correct']), color='g', label=f'Correct predictions - mean probability - {round(mean_prob_co, 2)}%')
		# Add incorrect predictions mean probability
		mean_prob_inco = np.mean(pred_prob_dict['incorrect'])
		plt.axvline(x=mean_prob_inco, color='r', label=f'Incorrect predictions - mean probability - {round(mean_prob_inco, 2)}%')
		if model_acc:
			plt.axvline(x=self.get_model_accuracy(pred_prob_dict), color='b', label=f'Model accuracy - {round(self.get_model_accuracy(pred_prob_dict), 2)}%')
		# Format y axis as a percentage (example: from 0.8 to 80%)
		plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

		# Add legend
		plt.legend(loc='upper right')
		plt.title(filename.replace('_', ' ').replace('.png', ' ').replace('std', 'std deviation'))
		plt.xlabel('Prediction probability (%)')
		plt.ylabel('Distribution')
		plt.savefig(self.output_dir / filename, bbox_inches='tight')
	
	def plot_scatter(self, filename, dict_mean, dict_std):
		def polyfit(x, y):
			a, b, c, d = np.polyfit(x, y, 3)
			fit_equation = lambda x: a*x**3 + b*x**2 + c*x + d
			x_fit = np.linspace(min(x), max(x), 1000)
			y_fit = fit_equation(x_fit)
			
			return x_fit, y_fit

		# Extract x and y coordinates from dictionaries
		x_correct = dict_mean['correct']
		y_correct = dict_std['correct']
			
		# Create scatter plot
		plt.figure(figsize=(10, 10))
		plt.scatter(x_correct, y_correct, alpha=0.8, c='lightgreen', label='Correct')

		# Add polyfit line
		x_fit, y_fit = polyfit(x_correct, y_correct)
		plt.plot(x_fit, y_fit, color='g', label='Correct - 3rd order polyfit')
		
		# Set axis range (Xmin, Xmax, Ymin, Ymax)
		plt.axis([50, 100, 0, 50])
		
		# Extract x and y coordinates from dictionaries
		x_incorrect = dict_mean['incorrect']
		y_incorrect = dict_std['incorrect']
		
		# Create scatter plot
		plt.scatter(x_incorrect, y_incorrect, alpha=0.8, c='lightcoral', label='Incorrect')
		
		# Add polyfit line
		x_fit, y_fit = polyfit(x_incorrect, y_incorrect)
		plt.plot(x_fit, y_fit, color='r',label='Incorrect - 3rd order polyfit')
		
		# Add labels and legend
		plt.xlabel('Prediction probability - mean')
		plt.ylabel('Prediction probability - standard deviation')
		plt.legend()

		plt.savefig(self.output_dir / filename, bbox_inches='tight')

	@staticmethod
	def get_model_accuracy(pred_dict):
		num_pred_co = len(pred_dict['correct'])
		num_pred_in = len(pred_dict['incorrect'])
		return (num_pred_co / (num_pred_co + num_pred_in)) * 100


def LeNet3d_VI():
	#data_dir = Path('F:\\master\\random_data\\50K_sample_1k_unique_slices')
	data_dir = Path('F:\\master\\LIDC\\nodules_16slices')
	output_dir = Path('F:\\master\\LIDC\\output_bayesian')
	models_dir = Path('C:\\Users\\pau_a\\Documents\\Python_scripts\\bayesian_convolutional_neural_network\\lidc\\bayesian\\models')

	exp = RunExperiment(
		train=True,
		n_batch=64,
		save_dir=models_dir / '20240504_LeNet3D_v2',
		num_networks=10,
	)
	exp.load_data(data_dir)
	exp.get_models(dropout=False)

	exp.test_model_ensemble(output_dir=output_dir / 'wo_dropout' / '20240503_LeNet3D_eval_w_eval_dataset_only')

def LeNet3d_VI_dropout():
	#data_dir = Path('F:\\master\\random_data\\50K_sample_1k_unique_slices')
	data_dir = Path('F:\\master\\LIDC\\nodules_16slices')
	output_dir = Path('F:\\master\\LIDC\\output_bayesian')
	models_dir = Path('C:\\Users\\pau_a\\Documents\\Python_scripts\\bayesian_convolutional_neural_network\\lidc\\bayesian\\models')

	exp = RunExperiment(
		train=True,
		n_batch=64,
		save_dir=models_dir / '20240504_LeNet3D_dropout',
		num_networks=10,
	)
	exp.load_data(data_dir)
	exp.get_models(dropout=True)

	exp.test_model_ensemble(output_dir=output_dir / 'w_dropout' / '20240504_LeNet3D_dropout' / 'lidc')

if __name__ == '__main__':
	LeNet3d_VI()
