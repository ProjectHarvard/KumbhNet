"""
Heatmap Generator Module

This module converts normalized 2D crowd density maps into visible heatmap images
and optionally overlays them on original video frames.
"""

import numpy as np
import cv2
import os
import matplotlib.cm as cm
from typing import Optional


def generate_heatmap(
    density_map: np.ndarray,
    frame: Optional[np.ndarray] = None,
    frame_number: int = 0,
    output_dir: str = "outputs/heatmaps",
    colormap_name: str = "jet",
    alpha: float = 0.5
) -> str:
    """
    Convert density map to heatmap image and optionally overlay on frame.
    
    This function:
    - Converts density map to color heatmap using matplotlib colormap
    - Optionally overlays heatmap on original frame with transparency
    - Saves result as PNG file with frame number in filename
    - Creates output directory if it does not exist
    
    Args:
        density_map: 2D numpy array (H, W) with values in range [0, 1]
        frame: Optional original RGB frame as numpy array (H, W, 3).
               If provided, heatmap will be overlaid on this frame.
        frame_number: Frame number to include in output filename. Defaults to 0.
        output_dir: Directory to save heatmap images. Defaults to "outputs/heatmaps".
        colormap_name: Name of matplotlib colormap to use. Defaults to "jet".
        alpha: Transparency level for overlay (0.0 to 1.0). Defaults to 0.5.
               Only used when frame is provided.
        
    Returns:
        Path to the saved heatmap image file.
    """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get colormap from matplotlib
    colormap = cm.get_cmap(colormap_name)
    
    # Normalize density map to [0, 1] range (should already be normalized, but ensure it)
    density_normalized = np.clip(density_map, 0.0, 1.0)
    
    # Apply colormap to density map
    # colormap returns RGBA values in range [0, 1]
    heatmap_rgba = colormap(density_normalized)
    
    # Convert to uint8 format (0-255) and remove alpha channel for now
    # Shape: (H, W, 4) -> (H, W, 3) RGB
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    
    # If frame is provided, overlay heatmap on frame
    if frame is not None:
        # Ensure frame is in correct format (uint8, RGB)
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 255)).astype(np.uint8)
        
        # Resize heatmap to match frame dimensions if needed
        frame_height, frame_width = frame.shape[:2]
        heatmap_height, heatmap_width = heatmap_rgb.shape[:2]
        
        if heatmap_height != frame_height or heatmap_width != frame_width:
            heatmap_rgb = cv2.resize(
                heatmap_rgb,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Overlay heatmap on frame with transparency
        # Convert both to float for blending, then back to uint8
        frame_float = frame.astype(np.float32)
        heatmap_float = heatmap_rgb.astype(np.float32)
        
        # Blend: result = frame * (1 - alpha) + heatmap * alpha
        blended = frame_float * (1.0 - alpha) + heatmap_float * alpha
        
        # Convert back to uint8
        result_image = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        # No frame provided, use heatmap only
        result_image = heatmap_rgb
    
    # Generate output filename with frame number
    filename = f"heatmap_frame_{frame_number:06d}.png"
    output_path = os.path.join(output_dir, filename)
    
    # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    
    # Save image as PNG
    cv2.imwrite(output_path, result_bgr)
    
    return output_path
