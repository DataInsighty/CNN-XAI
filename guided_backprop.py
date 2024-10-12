import torch
import torch.nn as nn

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.hooks = []

        # Register hooks for ReLU layers
        self._register_hooks()

    def _register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(self._relu_backward_hook)
                self.hooks.append(hook)

    def _relu_backward_hook(self, module, grad_in, grad_out):
        """
        Modify the gradients: Only allow positive gradients to flow backward.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, input_image, target_class=None):
        input_image = input_image.requires_grad_()

        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax()

        # Zero gradients
        self.model.zero_grad()

        # Create one-hot encoding for the target class and backward pass
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, target_class] = 1
        output.backward(gradient=one_hot_output)

        # Get the gradients from the input image
        guided_gradients = input_image.grad.data[0].cpu().numpy()

        # Normalize gradients for better visualization
        guided_gradients = (guided_gradients - guided_gradients.min()) / (guided_gradients.max() - guided_gradients.min())

        # Remove hooks to clean up
        for hook in self.hooks:
            hook.remove()

        return guided_gradients
