# Source (28/03/2024): https://jacobgil.github.io/pytorch-gradcam-book/HuggingFace.html

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Forward pass through the model
        return self.model(x, stochastic=True)


def category_name_to_index(model, category_name):
    """
    Translate the category name to the category index.
        Some models aren't trained on Imagenet but on even larger datasets,
        so we can't just assume that 761 will always be remote-control.

    """
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]


def run_grad_cam_on_image(
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        targets_for_gradcam: List[Callable],
        reshape_transform: Optional[Callable],
        input_tensor: torch.nn.Module,
        input_image: Image,
        method: Callable = GradCAM,
):
    """ Helper function to run GradCAM on an image and create a visualization.
        (note to myself: this is probably useful enough to move into the package)
        If several targets are passed in targets_for_gradcam,
        e.g different categories,
        a visualization for each of them will be created.

    """
    with method(model=ModelWrapper(model),
                target_layers=[target_layer],
                reshape_transform=reshape_transform) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image) / 255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1] // 2, visualization.shape[0] // 2))
            results.append(visualization)
        return np.hstack(results)


def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")