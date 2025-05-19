# 🎨 Neural Style Transfer with PyTorch

A Python implementation of **Neural Style Transfer (NST)** using PyTorch and a pre-trained VGG19 model. This algorithm merges the content of one image with the artistic style of another, creating visually striking results.

## Features
- **Content-Style Fusion**: Combines a content image (e.g., a photo) with a style image (e.g., a painting).
- **Customizable Layers**: Targets specific VGG19 layers for style (`conv1-4`) and content (`conv4`).
- **Optimized Workflow**: Uses L-BFGS optimizer for efficient style transfer.
- **GPU Support**: Automatically leverages CUDA if available.

## Technologies Used
- **PyTorch** (Deep learning framework)
- **TorchVision** (Pre-trained VGG19 model and image transformations)
- **Pillow (PIL)** (Image loading/saving)
- **CUDA** (GPU acceleration, optional)

## How It Works
1. **Image Preprocessing**:  
   - Resizes images to `256x256` and normalizes them for VGG19.
2. **Model Setup**:  
   - Loads pre-trained VGG19 and extracts feature maps from specified layers.
3. **Loss Calculation**:  
   - **Content Loss**: Preserves the structure of the content image.  
   - **Style Loss** (Gram Matrix): Captures texture/color patterns from the style image.  
4. **Optimization**:  
   - Iteratively updates the generated image using L-BFGS to minimize total loss.
  
## Output
Sample Input:
![Image](https://github.com/user-attachments/assets/51c7003d-d329-4724-b754-68b640fdc507)
![Image](https://github.com/user-attachments/assets/2fc37569-762b-4dcc-84c7-903e8116035f)
its Output:
![Image](https://github.com/user-attachments/assets/e01660ab-d4b9-4e62-a8f1-19ff2df1f423)

## Installation
```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
pip install torch torchvision pillow
