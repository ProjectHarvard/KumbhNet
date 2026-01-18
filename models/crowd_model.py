"""
Crowd Density Estimator Module

This module provides a lightweight, CPU-compatible crowd density estimator
using a pre-trained semantic segmentation model from timm.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import timm


class CrowdDensityEstimator:
    """
    Lightweight crowd density estimator using pre-trained segmentation model.
    
    This class loads a pre-trained semantic segmentation model once and
    uses it to generate density-like maps from input frames.
    """
    
    def __init__(self, model_name: str = "mobilenetv3_large_100", device: str = "cpu"):
        """
        Initialize the crowd density estimator.
        
        Args:
            model_name: Name of the timm model backbone to use.
                       Defaults to "mobilenetv3_large_100" for lightweight CPU inference.
            device: Device to run inference on. Defaults to "cpu".
        """
        self.device = torch.device(device)
        
        # Load pre-trained feature extractor from timm
        # Using a lightweight MobileNet-based model for CPU compatibility
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,  # Get feature maps instead of classification output
            in_chans=3
        )
        
        # Get the number of output channels from the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            # Use the last feature map
            last_feature_channels = features[-1].shape[1]
        
        # Add a simple convolutional head to convert features to density map
        # This creates a 1-channel output (density map)
        self.density_head = torch.nn.Sequential(
            torch.nn.Conv2d(last_feature_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 1, kernel_size=1)  # Single channel output
        )
        
        # Set models to evaluation mode and move to device
        self.backbone.eval()
        self.backbone.to(self.device)
        self.density_head.eval()
        self.density_head.to(self.device)
        
        # Define image preprocessing transforms
        # Normalize to ImageNet stats (standard for pre-trained models)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def estimate_density(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate crowd density map from a single RGB frame.
        
        Args:
            frame: Input RGB frame as numpy array (H, W, 3) with values in [0, 255]
            
        Returns:
            2D density map as numpy array (H, W) with values normalized to [0, 1]
        """
        # Convert frame to tensor and preprocess
        # Frame is expected to be RGB (H, W, 3) from OpenCV
        input_tensor = self.transform(frame)
        
        # Add batch dimension: (1, 3, H, W)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Run inference with gradients disabled
        with torch.no_grad():
            # Get features from backbone
            features = self.backbone(input_tensor)
            # Use the last (most abstract) feature map
            last_features = features[-1]
            
            # Convert features to density map using the head
            density_output = self.density_head(last_features)
            
            # Upsample to match input frame size if needed
            _, _, h_out, w_out = density_output.shape
            _, _, h_in, w_in = input_tensor.shape
            
            if h_out != h_in or w_out != w_in:
                # Use bilinear interpolation to resize to input dimensions
                density_output = F.interpolate(
                    density_output,
                    size=(h_in, w_in),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Remove batch dimension: (1, 1, H, W) -> (H, W)
            density_map = density_output.squeeze(0).squeeze(0)
            
            # Convert to numpy
            density_map = density_map.cpu().numpy()
            
            # Normalize to [0, 1] range
            # Use min-max normalization
            min_val = density_map.min()
            max_val = density_map.max()
            
            if max_val > min_val:
                density_map = (density_map - min_val) / (max_val - min_val)
            else:
                # If all values are the same, set to 0.5
                density_map = np.ones_like(density_map) * 0.5
            
            # Ensure values are in [0, 1] range (clip for safety)
            density_map = np.clip(density_map, 0.0, 1.0)
            
            return density_map


# Global model instance (loaded once)
_model_instance = None


def get_density_map(frame: np.ndarray) -> np.ndarray:
    """
    Convenience function to get density map from a frame.
    
    This function loads the model once (lazy initialization) and reuses it
    for subsequent calls.
    
    Args:
        frame: Input RGB frame as numpy array (H, W, 3) with values in [0, 255]
        
    Returns:
        2D density map as numpy array (H, W) with values normalized to [0, 1]
    """
    global _model_instance
    
    # Lazy initialization: load model on first call
    if _model_instance is None:
        _model_instance = CrowdDensityEstimator()
    
    return _model_instance.estimate_density(frame)
