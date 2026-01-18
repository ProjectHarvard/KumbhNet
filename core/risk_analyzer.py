"""
Risk Analyzer Module

This module analyzes 2D crowd density maps to determine stampede risk
using a grid-based thresholding approach.
"""

import numpy as np
from typing import Tuple


# Risk analysis constants
GRID_ROWS = 10
GRID_COLS = 10
DANGEROUS_DENSITY_THRESHOLD = 0.7
LOW_RISK_THRESHOLD = 0.2
MEDIUM_RISK_THRESHOLD = 0.5


def analyze_risk(density_map: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Analyze crowd density map to determine stampede risk.
    
    This function:
    - Divides the density map into a fixed grid (10x10 cells by default)
    - Computes average density per grid cell
    - Marks cells as dangerous if density exceeds threshold (0.7 by default)
    - Computes risk score as ratio of dangerous cells to total cells
    - Determines risk level (LOW, MEDIUM, HIGH) based on risk score
    
    Args:
        density_map: 2D numpy array (H, W) with values in range [0, 1]
        
    Returns:
        Tuple containing:
            - risk_level: String indicating risk level ("LOW", "MEDIUM", or "HIGH")
            - risk_score: Float in range [0, 1] representing proportion of dangerous cells
            - risk_mask: 2D boolean numpy array (H, W) indicating dangerous regions
                        True where density exceeds threshold, False otherwise
    """
    # Handle edge cases: empty or invalid input
    if density_map is None or density_map.size == 0:
        return "LOW", 0.0, np.array([], dtype=bool)
    
    # Ensure density map is 2D
    if density_map.ndim != 2:
        raise ValueError(f"Expected 2D density map, got {density_map.ndim}D array")
    
    # Get dimensions
    height, width = density_map.shape
    
    # Handle very small images
    if height < GRID_ROWS or width < GRID_COLS:
        # For small images, use the entire map as a single cell
        avg_density = np.mean(density_map)
        is_dangerous = avg_density > DANGEROUS_DENSITY_THRESHOLD
        risk_score = 1.0 if is_dangerous else 0.0
        risk_level = _determine_risk_level(risk_score)
        risk_mask = np.full_like(density_map, is_dangerous, dtype=bool)
        return risk_level, risk_score, risk_mask
    
    # Divide density map into grid cells
    # Calculate cell dimensions
    cell_height = height // GRID_ROWS
    cell_width = width // GRID_COLS
    
    # Initialize grid for storing average densities
    grid_densities = np.zeros((GRID_ROWS, GRID_COLS))
    
    # Compute average density for each grid cell
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            # Define cell boundaries
            row_start = i * cell_height
            row_end = (i + 1) * cell_height if i < GRID_ROWS - 1 else height
            col_start = j * cell_width
            col_end = (j + 1) * cell_width if j < GRID_COLS - 1 else width
            
            # Extract cell region
            cell_region = density_map[row_start:row_end, col_start:col_end]
            
            # Compute average density for this cell
            grid_densities[i, j] = np.mean(cell_region)
    
    # Mark dangerous cells (where average density exceeds threshold)
    dangerous_cells = grid_densities > DANGEROUS_DENSITY_THRESHOLD
    
    # Compute risk score: proportion of dangerous cells
    total_cells = GRID_ROWS * GRID_COLS
    num_dangerous_cells = np.sum(dangerous_cells)
    risk_score = num_dangerous_cells / total_cells
    
    # Determine risk level based on risk score
    risk_level = _determine_risk_level(risk_score)
    
    # Create risk mask: boolean array indicating dangerous regions
    # Map grid-based dangerous cells back to full resolution
    risk_mask = np.zeros_like(density_map, dtype=bool)
    
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            if dangerous_cells[i, j]:
                # Mark all pixels in this cell as dangerous
                row_start = i * cell_height
                row_end = (i + 1) * cell_height if i < GRID_ROWS - 1 else height
                col_start = j * cell_width
                col_end = (j + 1) * cell_width if j < GRID_COLS - 1 else width
                
                risk_mask[row_start:row_end, col_start:col_end] = True
    
    return risk_level, risk_score, risk_mask


def _determine_risk_level(risk_score: float) -> str:
    """
    Determine risk level based on risk score.
    
    Args:
        risk_score: Risk score in range [0, 1]
        
    Returns:
        Risk level string: "LOW", "MEDIUM", or "HIGH"
    """
    if risk_score < LOW_RISK_THRESHOLD:
        return "LOW"
    elif risk_score < MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"
