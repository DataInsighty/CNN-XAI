import torch
import numpy as np
from guided_backprop import GuidedBackprop
from explainer_gradcam import GradCAM  # Assuming GradCAM is implemented

class GuidedGradCAM:
    def __init__(self, model, target_layer):
        self.gradcam = GradCAM(model, target_layer)
        self.guided_backprop = GuidedBackprop(model)

    def __call__(self, input_image, target_class=None):
        # Generate Grad-CAM
        grad_cam_output = self.gradcam(input_image)

        # Normalize the Grad-CAM output
        grad_cam_output = (grad_cam_output - grad_cam_output.min()) / (grad_cam_output.max() - grad_cam_output.min())

        # Generate Guided Backpropagation
        guided_backprop_output = self.guided_backprop(input_image, target_class)

        # Combine Grad-CAM and Guided Backpropagation
        guided_gradcam_output = np.multiply(grad_cam_output, guided_backprop_output)

        # Normalize the combined output for visualization
        guided_gradcam_output = (guided_gradcam_output - guided_gradcam_output.min()) / (guided_gradcam_output.max() - guided_gradcam_output.min())

        return guided_gradcam_output
