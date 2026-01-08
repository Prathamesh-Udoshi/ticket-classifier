"""Confidence score utilities for classification."""

import numpy as np


def get_confidence_score(probabilities: np.ndarray) -> float:
    """
    Extract confidence score from probability array.
    
    Confidence is the maximum probability value, which represents
    the model's certainty in the predicted class.
    
    Args:
        probabilities: Array of class probabilities from model
        
    Returns:
        Confidence score as float between 0 and 1
    """
    if probabilities is None or len(probabilities) == 0:
        return 0.0
    
    # Confidence is the maximum probability
    confidence = float(np.max(probabilities))
    
    # Ensure it's between 0 and 1
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence


def format_confidence(confidence: float, decimals: int = 2) -> str:
    """
    Format confidence score as percentage string.
    
    Args:
        confidence: Confidence score (0-1)
        decimals: Number of decimal places
        
    Returns:
        Formatted string (e.g., "85.50%")
    """
    percentage = confidence * 100
    return f"{percentage:.{decimals}f}%"

