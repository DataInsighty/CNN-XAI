import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNN_BT
from explainer_gradcam import GradCAM, plot_grad_cam  # Grad-CAM functions
from guided_backprop import GuidedBackprop  # Import Guided Backpropagation
from guided_gradcam import GuidedGradCAM  # Import Guided Grad-CAM
import matplotlib.pyplot as plt
import numpy as np

# Define the parameters needed for the CNN_BT model
params = {
    "shape_in": (3, 256, 256),  # Input shape: (channels, height, width)
    "initial_filters": 8,        # Initial number of filters
    "num_fc1": 100,              # Number of neurons in the first fully connected layer
    "dropout_rate": 0.25,        # Dropout rate for regularization
    "num_classes": 2             # Number of output classes
}

# Define a mapping from numeric labels to human-readable labels
label_mapping = {0: "Brain Tumor", 1: "Healthy"}  # This maps the output index to a label

# Initialize the model
model = CNN_BT(params)

# Move the model to the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define image transformations (resize the image to the model's expected input size and convert to tensor)
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# Function to make predictions
def predict(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert grayscale images to RGB
    
    image = transform(image).unsqueeze(0)  # Transform and add batch dimension
    image = image.to(device)  # Move the image to the same device as the model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation (not needed for inference)
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)  # Get the index of the highest score
    return predicted.item(), image  # Return both prediction and the transformed image

# Streamlit app
st.title("Brain Tumor Classification with Explainable AI")

# Initialize session state variables if not present
if 'predicted_image_tensor' not in st.session_state:
    st.session_state['predicted_image_tensor'] = None
if 'label_name' not in st.session_state:
    st.session_state['label_name'] = None
if 'original_image' not in st.session_state:
    st.session_state['original_image'] = None
if 'grad_cam_fig' not in st.session_state:
    st.session_state['grad_cam_fig'] = None
if 'guided_backprop_fig' not in st.session_state:
    st.session_state['guided_backprop_fig'] = None
if 'guided_gradcam_fig' not in st.session_state:
    st.session_state['guided_gradcam_fig'] = None

# File uploader for images
uploaded_file = st.file_uploader("Choose an image of a brain scan...", type=["jpg", "png", "jpeg", "tiff"])

# Add a submit button for classification
if st.button('Submit for Classification'):
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        st.session_state['original_image'] = image  # Store the original image

        # Display the uploaded image
        st.image(image, caption='Uploaded Brain Scan', use_column_width=True)

        # Classify the image using the model
        st.write("Classifying...")
        label_index, predicted_image_tensor = predict(image)  # Get the predicted index (0 or 1) and transformed image tensor

        # Store the predicted image tensor and label name in session state
        st.session_state['predicted_image_tensor'] = predicted_image_tensor
        st.session_state['label_name'] = label_mapping[label_index]

        # Display the predicted label name
        st.write(f"**Predicted Label:** {st.session_state['label_name']}")
    else:
        st.write("Please upload an image first.")

# Grad-CAM Button
if st.button('Explain with Grad-CAM'):
    if st.session_state['predicted_image_tensor'] is not None:
        st.write(f"Generating Grad-CAM for label: {st.session_state['label_name']}")
        grad_cam = GradCAM(model, model.conv4)  # Assuming conv4 is the target layer
        grad_cam_output = grad_cam(st.session_state['predicted_image_tensor'])  # Generate Grad-CAM heatmap
        fig = plot_grad_cam(st.session_state['predicted_image_tensor'].squeeze(0), grad_cam_output)
        st.session_state['grad_cam_fig'] = fig  # Store figure in session state

        st.write("""
        **Grad-CAM Explanation**: 
        - Grad-CAM generates a heatmap that highlights the regions in the brain scan that had the most influence on the model's prediction.
        - Red areas indicate where the model focused the most, helping to understand its decision-making process.
        """)

# Display the Grad-CAM image below the button
if st.session_state['grad_cam_fig'] is not None:
    st.write("### Grad-CAM Result")
    st.pyplot(st.session_state['grad_cam_fig'])

# Guided Backpropagation Button
if st.button('Explain with Guided Backpropagation'):
    if st.session_state['predicted_image_tensor'] is not None:
        st.write(f"Generating Guided Backpropagation for label: {st.session_state['label_name']}")
        guided_backprop = GuidedBackprop(model)
        guided_backprop_output = guided_backprop(st.session_state['predicted_image_tensor'])  # Generate guided backprop output
        
        # Plot the guided backprop output and normalize
        guided_backprop_output = np.transpose(guided_backprop_output, (1, 2, 0))  # Change shape for plotting
        fig, ax = plt.subplots(figsize=(10, 10))  # Create a large figure
        ax.imshow(guided_backprop_output)
        st.session_state['guided_backprop_fig'] = fig  # Store figure in session state

        st.write("""
        **Guided Backpropagation Explanation**: 
        - This technique shows the pixel-level gradients, revealing which specific pixels in the image contributed the most to the model’s prediction.
        - It provides more detailed, high-resolution feedback compared to Grad-CAM, showing finer visual details of how the model processes the input.
        """)

# Display the Guided Backpropagation image below the button
if st.session_state['guided_backprop_fig'] is not None:
    st.write("### Guided Backpropagation Result")
    st.pyplot(st.session_state['guided_backprop_fig'])

# Guided Grad-CAM Button
if st.button('Explain with Guided Grad-CAM'):
    if st.session_state['predicted_image_tensor'] is not None:
        st.write(f"Generating Guided Grad-CAM for label: {st.session_state['label_name']}")
        guided_gradcam = GuidedGradCAM(model, model.conv4)  # Assuming conv4 is the target layer
        guided_gradcam_output = guided_gradcam(st.session_state['predicted_image_tensor'])  # Generate guided grad-cam output

        # Plot the guided grad-cam output and normalize
        guided_gradcam_output = np.transpose(guided_gradcam_output, (1, 2, 0))  # Change shape for plotting
        fig, ax = plt.subplots(figsize=(10, 10))  # Create a large figure
        ax.imshow(guided_gradcam_output)
        st.session_state['guided_gradcam_fig'] = fig  # Store figure in session state

        st.write("""
        **Guided Grad-CAM Explanation**: 
        - This method combines the high-level focus regions from Grad-CAM with the fine details from Guided Backpropagation.
        - It provides a more comprehensive view by showing both the important regions and the detailed gradients that affected the model’s decision.
        """)

# Display the Guided Grad-CAM image below the button
if st.session_state['guided_gradcam_fig'] is not None:
    st.write("### Guided Grad-CAM Result")
    st.pyplot(st.session_state['guided_gradcam_fig'])

# Compare All Images in a Row
if st.button('Compare All Images'):
    if st.session_state['original_image'] is not None and \
       st.session_state['grad_cam_fig'] is not None and \
       st.session_state['guided_backprop_fig'] is not None and \
       st.session_state['guided_gradcam_fig'] is not None:
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Display Original Image
        with col1:
            st.image(st.session_state['original_image'], caption='Original Image', use_column_width=True)
        
        # Display Grad-CAM Image
        with col2:
            st.pyplot(st.session_state['grad_cam_fig'])

        # Display Guided Backpropagation Image
        with col3:
            st.pyplot(st.session_state['guided_backprop_fig'])
        
        # Display Guided Grad-CAM Image
        with col4:
            st.pyplot(st.session_state['guided_gradcam_fig'])
    else:
        st.write("Please generate all the images before comparing them.")

# Always display the predicted label
if st.session_state['label_name']:
    st.write(f"**Predicted Label:** {st.session_state['label_name']}")

# Clear All Images Button
if st.button('Clear All Images'):
    st.session_state['grad_cam_fig'] = None
    st.session_state['guided_backprop_fig'] = None
    st.session_state['guided_gradcam_fig'] = None
    st.session_state['original_image'] = None
    st.write("All images cleared.")

# Disclaimer
st.write("""
---
**Disclaimer:** The images and predictions generated by this application are AI-based and intended for educational purposes only. 
They should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for a proper diagnosis and treatment plan.
""")
