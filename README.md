# Neural Style Transfer with Multi-Input Conditioning

This repository contains a Jupyter notebook implementing **Neural Style Transfer** using a Convolutional Neural Network (CNN) based on VGG19, with additional support for a Vision Transformer (ViT) extension. The project enables blending content and style images with a threshold-based approach to control the style intensity, and includes a web application for interactive use.

## Overview

The notebook (`Neural_Style_Transfer_Divided.ipynb`) implements:
- **Neural Style Transfer**: Transfers the artistic style of one image to the content of another using a pre-trained VGG19 model.
- **Threshold-based Style Blending**: Adjusts the balance between content and style using a customizable threshold.
- **Dataset Preparation**: Processes content and style images for training and evaluation.
- **Evaluation and Visualization**: Generates and visualizes stylized images.
- **Vision Transformer Extension**: Incorporates a ViT-based feature blender for advanced feature extraction.
- **Web Application**: A Gradio-based interface for interactive style transfer.

## Features

- **CNN-based Style Transfer**: Uses VGG19 to extract features and a custom decoder to generate stylized images.
- **Threshold Control**: Allows dynamic adjustment of style influence via a threshold parameter.
- **Vision Transformer Integration**: Optional ViT module for feature blending, leveraging `google/vit-base-patch16-224`.
- **Interactive Web App**: Built with Gradio, enabling users to upload content/style images and adjust the style threshold.
- **Training and Evaluation**: Supports training with content/style datasets and evaluation with visualizations.



## Usage

1. **Prepare Datasets**:
   - Place content images in `data/content/`.
   - Place style images in `data/style/`.
   - Images should be in a format readable by PIL (e.g., JPEG, PNG).

2. **Run the Notebook**:
   - Open `Neural_Style_Transfer_Divided.ipynb` in Jupyter.
   - Execute cells sequentially to:
     - Load dependencies and define models.
     - Prepare the dataset and data loaders.
     - Train the model (50 epochs by default).
     - Save the trained model as `style_transfer_model.pth`.
     - Evaluate the model and generate visualizations.
     - Launch the Gradio web app for interactive style transfer.

3. **Train the Model**:
   - The dataset is split into 70% training, 15% validation, and 15% testing.
   - Training uses Adam optimizer with MSE loss for content and Gram matrix-based loss for style, modulated by the threshold.

4. **Use the Web App**:
   - The final cell launches a Gradio interface.
   - Upload a content image, a style image, and adjust the style threshold (0 to 1).
   - The app outputs the stylized image.

## Key Components

- **VGGFeatures**: Extracts features from VGG19 for content and style images.
- **Decoder**: Generates stylized images from blended features.
- **StyleTransferNet**: Combines VGG encoder and decoder, blending features based on the threshold.
- **ViTFeatureBlender**: Optional module using a Vision Transformer for feature extraction and blending.
- **Training Loop**: Optimizes the model with content and style losses, saving preview images periodically.
- **Evaluation**: Visualizes stylized outputs with matplotlib.
- **Gradio Interface**: Provides a user-friendly web interface for style transfer.

## Notes

- **Pre-trained Models**: The notebook uses `VGG19_Weights.IMAGENET1K_V1` and `google/vit-base-patch16-224`. Ensure internet access for downloading or provide local weights.
- **Performance**: Training is computationally intensive; a CUDA-enabled GPU is recommended.
- **Customization**: Adjust `IMAGE_SIZE`, `BATCH_SIZE`, and `EPOCHS` in the notebook to suit your needs.
- **Warnings**: The notebook may show deprecation warnings for `pretrained` in torchvision; these can be ignored or updated to use `weights`.

## Example Output

During training, preview images are saved as `preview_epochX.jpg` every 5 epochs. The evaluation step generates up to 10 stylized images with corresponding thresholds. The Gradio app allows real-time experimentation with custom inputs.

## Future Improvements

- Optimize training speed with mixed precision.
- Add support for multiple style images per content image.
- Enhance the ViT module with a trainable decoder.
- Include additional evaluation metrics (e.g., SSIM, perceptual loss).



For issues or contributions, please contact me basemhesham200318@gmail.com or open a pull request or issue on this GitHub repository.
