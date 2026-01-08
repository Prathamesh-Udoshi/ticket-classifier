"""Text preprocessing utilities for ticket classification."""

import re
import string


def clean_text(text: str) -> str:
    """
    Clean and normalize ticket text for classification.
    
    Performs basic text cleaning:
    - Converts to lowercase
    - Removes extra whitespace
    - Removes special characters (keeps alphanumeric and basic punctuation)
    - Normalizes spacing
    
    Args:
        text: Raw ticket text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize(text: str) -> list:
    """
    Simple tokenization of text.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens (words)
    """
    if not text:
        return []
    
    # Split on whitespace and filter empty strings
    tokens = [token.strip() for token in text.split() if token.strip()]
    return tokens

