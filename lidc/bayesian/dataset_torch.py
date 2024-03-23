import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from scipy.ndimage import rotate
import random
import matplotlib.pyplot as plt


class LidcNoduleDataset(Dataset):
    def __init__(self, lidc_path, num_slices, transform=None, augment=None, augmentation_prob=0.8):
        self.lidc_path = lidc_path
        self.num_slices = num_slices
        self.transform = transform
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        self.images = []
        self.diagnostics = []
        self.load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        diag = self.diagnostics[idx]

        if self.augment and self.should_augment(diag):
            image = self.augment(image)

        if self.transform:
            image = self.transform(image)

        return image, diag

    def load_data(self):
        nodules_path = list(self.lidc_path.iterdir())
        for nodule_path in nodules_path:
            images_path = list(nodule_path.iterdir())
            images_path = images_path[
                          int(len(images_path) / 2) - int(self.num_slices / 2):int(len(images_path) / 2) + int(
                              self.num_slices / 2)]
            image_nodule = []
            for image_path in images_path:
                image_slice = np.array(Image.open(image_path)) / 255.0
                image_nodule.append(image_slice)
            image_nodule = np.array(image_nodule)

            diag = int(nodule_path.name.split('_')[5])
            if image_nodule.shape == (self.num_slices, 64, 64) and diag != 3:
                image_nodule = np.expand_dims(image_nodule, axis=0)
                self.images.append(image_nodule)
                if diag < 3:
                    self.diagnostics.append(0)
                else:
                    self.diagnostics.append(1)
            elif image_nodule.shape != (self.num_slices, 64, 64):
                print(
                    f'[ERROR] {nodule_path.name} has shape {image_nodule.shape}, but expected shape is: ({self.num_slices}, 64, 64)')

    def should_augment(self, label):
        if label == 1:
            # Apply augmentation with probability 1 for label 1
            return True
        elif label == 0:
            # Apply augmentation with probability self.augmentation_prob for label 0
            return random.random() < self.augmentation_prob
        else:
            return False


# Define data augmentation function
def rotate_image(image):
    angles = [-20, -10, -5, 5, 10, 20]
    angle = random.choice(angles)
    return rotate(image.transpose(), angle, reshape=False).transpose()


# Define custom transformation to preserve original shape
class ToTensorWithOriginalShape(object):
    def __call__(self, image):
        # Convert to tensor
        image = torch.from_numpy(image)
        return image.float()


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of X CT slices"""
    data = np.array(data)
    data = np.reshape(data, (6, 64, 64))
    data = np.reshape(data, (num_rows, num_columns, width, height))
    print(f'data shape: {data.shape}')
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i, j], cmap="gray")  # Corrected indexing
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


if __name__ == '__main__':
    # Example usage
    lidc_path = Path('F:\\master\\manifest-1600709154662\\nodules_16slices')
    num_slices = 6

    # Create dataset instance with augmentation
    dataset = LidcNoduleDataset(lidc_path, num_slices, transform=ToTensorWithOriginalShape(), augment=rotate_image)

    # Example access to data
    image, label = dataset[0]

    print(f'label[0] = {label}')
    for i, data in enumerate(dataset):
        shape = np.array(data[0]).shape
        if shape != (1, 6, 64, 64):
            print(f'[ERROR] data point {i} has shape {shape}, but expected shape is: (1, 6, 64, 64)')
    # Visualize montage of slices.
    plot_slices(
        num_rows=2,
        num_columns=3,
        width=64,
        height=64,
        data=np.squeeze(image, axis=-1)
    )
