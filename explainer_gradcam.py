import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import img_as_float

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

        # Hook for gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_image):
        # Forward pass and gradient calculation
        input_image = input_image.requires_grad_(True)  # Enable gradients for input image

        # Forward pass
        output = self.model(input_image)
        class_idx = output.argmax().item()  # Get predicted class index

        # Zero gradients and backward pass
        self.model.zero_grad()
        output[:, class_idx].backward()

        # Compute weights from gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        grad_cam = (weights * self.activation).sum(dim=1).squeeze(0)

        # Apply ReLU to keep only positive influences
        grad_cam = torch.clamp(grad_cam, min=0)

        # Resize the heatmap to match the input image size
        grad_cam = torch.nn.functional.interpolate(
            grad_cam.unsqueeze(0).unsqueeze(0), size=(input_image.size(2), input_image.size(3)), mode='bilinear'
        ).squeeze()

        return grad_cam.cpu().detach().numpy()

# Define plot_grad_cam function
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage import img_as_float

def plot_grad_cam(input_image, grad_cam_output, cmap='jet', alpha=0.5):
    """
    Function to plot the original image with the Grad-CAM heatmap overlay.
    
    Args:
    - input_image: The original image (PyTorch tensor or NumPy array).
    - grad_cam_output: The Grad-CAM output (heatmap as a NumPy array).
    - cmap: Colormap to use for the heatmap (default is 'jet').
    - alpha: Transparency level for overlaying heatmap (default is 0.5).
    
    Returns:
    - fig: Matplotlib figure with Grad-CAM heatmap overlaid.
    """

    # Convert input_image to NumPy if it's a PyTorch tensor
    if isinstance(input_image, torch.Tensor):
        input_image = input_image.permute(1, 2, 0).cpu().detach().numpy()  # Detach before calling .numpy()

    # Normalize the input image to [0, 1]
    input_image = img_as_float(input_image)

    # Resize the grad_cam_output to match the input image size
    grad_cam_resized = resize(grad_cam_output, (input_image.shape[0], input_image.shape[1]), preserve_range=True)

    # Create a figure
    fig, ax = plt.subplots()

    # Display the original image
    ax.imshow(input_image)

    # Overlay the Grad-CAM heatmap
    heatmap = ax.imshow(grad_cam_resized, cmap=cmap, alpha=alpha)

    # Add color bar for the heatmap
    plt.colorbar(heatmap, label="Importance")

    # Set title and turn off axis
    plt.title("Grad-CAM")
    plt.axis('off')

    return fig

