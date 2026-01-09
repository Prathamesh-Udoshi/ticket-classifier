# Support Ticket Auto-Classification Module

A production-ready Python module for automatically classifying customer support tickets into predefined categories, with confidence scores and priority assignment.

## Problem Statement

Customer support teams receive a high volume of tickets daily. Manually categorizing and prioritizing these tickets is time-consuming and can lead to inconsistent routing. This module automates the classification process, allowing support teams to:

- Automatically route tickets to the appropriate team
- Prioritize urgent issues
- Track classification confidence for quality assurance
- Scale support operations efficiently

## Approach

### Machine Learning Model
- **Algorithm**: TF-IDF (Term Frequency-Inverse Document Frequency) + Multinomial Naive Bayes
- **Rationale**: 
  - Fast inference suitable for real-time classification
  - Explainable predictions (probability scores for each category)
  - Works well with small to medium-sized datasets
  - No GPU required, easy to deploy
  - Naive Bayes is particularly effective for text classification tasks
  - Handles high-dimensional sparse data (like TF-IDF vectors) efficiently

### Priority Assignment
- **Method**: Rule-based logic (not ML-based)
- **Rules**:
  - **HIGH**: Contains urgent keywords (refund, error, failed, not working, etc.)
  - **MEDIUM**: Default priority for standard tickets
  - **LOW**: Positive sentiment detected (thank you, great, excellent, etc.)

## Categories

The module classifies tickets into four fixed categories:

1. **Billing** - Payment issues, refunds, charges, subscriptions
2. **Technical** - System errors, bugs, technical problems, outages
3. **Account** - Account management, profile updates, access issues
4. **Feedback** - Positive feedback, suggestions, general comments

## Project Structure

```
ticket_classifier/
├── app.py                 # Streamlit web interface
├── classifier/
│   ├── __init__.py
│   ├── model.py          # ML model implementation
│   └── labels.py         # Category and priority definitions
├── preprocess/
│   ├── __init__.py
│   └── clean_text.py     # Text preprocessing utilities
├── utils/
│   ├── __init__.py
│   ├── confidence.py     # Confidence score utilities
│   └── priority.py       # Priority assignment logic
├── data/
│   └── sample_tickets.csv  # Synthetic training dataset
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Streamlit Web Interface (Recommended)

```bash
streamlit run app.py
```

The interface will open in your browser at `http://localhost:8501`

### Programmatic Usage

```python
from classifier.model import train_model_from_csv
from preprocess.clean_text import clean_text
from utils.priority import assign_priority

# Load and train model
classifier = train_model_from_csv("data/sample_tickets.csv")

# Classify a ticket
ticket_text = "I was charged twice for my subscription. Can I get a refund?"
cleaned_text = clean_text(ticket_text)
category, confidence, probabilities = classifier.predict(cleaned_text)
priority = assign_priority(ticket_text)

print(f"Category: {category}")
print(f"Confidence: {confidence:.2%}")
print(f"Priority: {priority}")
```

## Model Training

The model is trained on the synthetic dataset in `data/sample_tickets.csv`. To retrain with your own data:

1. Prepare a CSV file with columns: `text`, `label`
2. Ensure labels match the categories: Billing, Technical, Account, Feedback
3. Update the path in `app.py` or use `train_model_from_csv()` directly

```python
from classifier.model import train_model_from_csv

classifier = train_model_from_csv("path/to/your/data.csv")
classifier.save("models/trained_model.pkl")  # Save for later use
```

## Limitations

1. **Training Dataset Size**: The model performance depends on the quality and quantity of training data. Adding more diverse, real-world examples will improve accuracy and generalization.

2. **Fixed Categories**: Categories are hardcoded. Adding new categories requires retraining and code changes.

3. **English Only**: The model is optimized for English text. Multilingual support would require additional preprocessing and potentially different models.

4. **Simple Priority Logic**: Priority assignment is rule-based and may not capture all edge cases. Consider ML-based priority prediction for more nuanced results.

5. **No Context**: The model doesn't consider ticket metadata (customer tier, ticket history, etc.) which could improve classification.

6. **TF-IDF + Naive Bayes Limitations**: While fast and explainable, TF-IDF may not capture semantic relationships as well as modern embeddings (Word2Vec, BERT, etc.). Naive Bayes assumes feature independence, which may not always hold true for text.

## Next Steps

### Short-term Improvements
1. **Model Retraining**: Collect real ticket data and retrain the model for better accuracy
2. **Model Persistence**: Save trained models to disk and load them in production (already implemented)
3. **Evaluation Metrics**: Add cross-validation and classification reports (precision, recall, F1-score)
4. **Confidence Thresholds**: Implement minimum confidence thresholds for auto-routing vs. manual review
5. **Model Comparison**: Experiment with different classifiers (Logistic Regression, SVM, Random Forest) to find optimal performance

### Medium-term Enhancements
1. **Monitoring**: Add logging and metrics tracking for production monitoring
   - Track prediction confidence distributions
   - Monitor category distribution shifts
   - Alert on low-confidence predictions
2. **A/B Testing**: Compare model performance against manual classification
3. **Feedback Loop**: Allow support agents to correct misclassifications and retrain periodically
4. **Batch Processing**: Add support for classifying multiple tickets at once

### Long-term Considerations
1. **Multilingual Support**: Extend to support multiple languages
   - Language detection
   - Language-specific models or multilingual embeddings
2. **Advanced ML Models**: Consider upgrading to:
   - Transformer-based models (BERT, DistilBERT) for better semantic understanding
   - Ensemble methods combining Naive Bayes with other classifiers for improved accuracy
   - Fine-tuned Naive Bayes with different smoothing parameters (alpha tuning)
3. **Contextual Features**: Incorporate ticket metadata:
   - Customer tier/segment
   - Historical ticket patterns
   - Time-based features (business hours, holidays)
4. **ML-based Priority**: Replace rule-based priority with a trained classifier
5. **API Integration**: Create REST API for integration with existing ticketing systems

## Dependencies

- `scikit-learn` - Machine learning (TF-IDF, Multinomial Naive Bayes)
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `streamlit` - Web interface (optional, for app.py)

See `requirements.txt` for specific versions.

## License

Internal use only - for integration into existing support ticket system.

## Contact

For questions or issues, contact the ML engineering team.

