"""
Video Reader Module

This module is responsible for reading video files and yielding frames
for further processing. It contains no machine learning or analysis logic.
"""

import cv2
import os
from typing import Generator, Tuple


def read_video_frames(video_path: str = "data/videos/crowd_input.mp4") -> Generator[Tuple, None, None]:
    """
    Read video file and yield processed frames.
    
    This function:
    - Opens the video file using OpenCV
    - Reads frames sequentially
    - Processes every 3rd frame only (frame skipping for efficiency)
    - Resizes frames to 640px width while maintaining aspect ratio
    - Yields each processed frame
    
    Args:
        video_path: Path to the video file. Defaults to "data/videos/crowd_input.mp4"
        
    Yields:
        Tuple containing:
            - frame: The processed frame (numpy array)
            - frame_number: The original frame number in the video
            
    Raises:
        FileNotFoundError: If the video file does not exist
        ValueError: If the video file cannot be opened or is invalid
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video was opened successfully
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        frame_count = 0
        target_width = 640
        
        while True:
            ret, frame = cap.read()
            
            # Break if no more frames
            if not ret:
                break
            
            # Process every 3rd frame only (frame skipping)
            if frame_count % 3 == 0:
                # Get original dimensions
                original_height, original_width = frame.shape[:2]
                
                # Calculate aspect ratio
                aspect_ratio = original_height / original_width
                
                # Calculate new height to maintain aspect ratio
                target_height = int(target_width * aspect_ratio)
                
                # Resize frame to target width while maintaining aspect ratio
                resized_frame = cv2.resize(frame, (target_width, target_height))
                
                # Yield the processed frame along with its original frame number
                yield (resized_frame, frame_count)
            
            frame_count += 1
            
    finally:
        # Always release the video capture object
        cap.release()
