"""Priority assignment logic for tickets."""

from typing import List
from classifier.labels import HIGH_PRIORITY_KEYWORDS, POSITIVE_SENTIMENT_WORDS, PRIORITY_LEVELS


def assign_priority(text: str) -> str:
    """
    Assign priority level to a ticket based on rule-based logic.
    
    Priority rules:
    - HIGH: Contains urgent keywords (refund, error, failed, etc.)
    - LOW: Contains positive sentiment words and no urgent keywords
    - MEDIUM: Default case
    
    Args:
        text: Ticket text string (should be lowercase for consistency)
        
    Returns:
        Priority level: "Low", "Medium", or "High"
    """
    if not text:
        return "Medium"
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check for high priority keywords
    has_urgent_keywords = any(keyword in text_lower for keyword in HIGH_PRIORITY_KEYWORDS)
    
    # Check for positive sentiment
    has_positive_sentiment = any(word in text_lower for word in POSITIVE_SENTIMENT_WORDS)
    
    # Priority logic
    if has_urgent_keywords:
        return "High"
    elif has_positive_sentiment and not has_urgent_keywords:
        return "Low"
    else:
        return "Medium"

