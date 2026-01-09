"""Label definitions for ticket classification."""

# Fixed categories for ticket classification
CATEGORIES = [
    "Account",
    "Billing",
    "Feedback",
    "Technical"
]

# Priority levels
PRIORITY_LEVELS = ["Low", "Medium", "High"]

# High priority keywords (case-insensitive matching)
HIGH_PRIORITY_KEYWORDS = [
    "urgent",
    "refund",
    "charged twice",
    "not working",
    "error",
    "failed",
    "broken",
    "down",
    "outage",
    "critical",
    "immediately",
    "asap"
]

# Positive sentiment indicators (for LOW priority)
POSITIVE_SENTIMENT_WORDS = [
    "thank",
    "thanks",
    "great",
    "excellent",
    "wonderful",
    "appreciate",
    "helpful",
    "love",
    "amazing",
    "perfect"
]

