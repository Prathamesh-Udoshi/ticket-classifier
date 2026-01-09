"""ML model implementation for ticket classification."""

import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from classifier.labels import CATEGORIES
from preprocess.clean_text import clean_text


class TicketClassifier:
    """
    Ticket classification model using TF-IDF + Logistic Regression.
    
    This class encapsulates the training and prediction logic for
    classifying support tickets into predefined categories.
    """
    
    def __init__(self):
        """Initialize the classifier with empty pipeline."""
        self.pipeline = None
        self.is_trained = False
    
    def _create_pipeline(self) -> Pipeline:
        """
        Create sklearn pipeline with TF-IDF vectorizer and Logistic Regression.
        
        Returns:
            Configured sklearn Pipeline
        """
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # Unigrams and bigrams
            min_df=1,
            max_df=0.9,
            stop_words='english',
            sublinear_tf=True
        )
        
        '''classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'
        )'''
            
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB(alpha=1.0)
        
        
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def train(self, X: list, y: list) -> None:
        """
        Train the classification model.
        
        Args:
            X: List of ticket text strings
            y: List of category labels (must match CATEGORIES)
        """
        if not X or not y:
            raise ValueError("Training data cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        # Normalize training text using the same cleaning as prediction
        cleaned_X = [clean_text(text) for text in X]

        # Create pipeline if not exists
        if self.pipeline is None:
            self.pipeline = self._create_pipeline()
        
        # Train the model
        self.pipeline.fit(cleaned_X, y)
        self.is_trained = True
    
    def predict(self, text: str) -> Tuple[str, float, np.ndarray]:
        """
        Predict category for a single ticket text.
        
        Args:
            text: Ticket text string
            
        Returns:
            Tuple of (predicted_category, confidence_score, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        # Clean text with the same preprocessing used for training
        cleaned_text = clean_text(text)

        # Get probabilities for all classes
        probabilities = self.pipeline.predict_proba([cleaned_text])[0]
        
        # Get predicted class index
        predicted_idx = np.argmax(probabilities)
        # Use pipeline-learned class ordering to avoid mismatches
        predicted_category = self.pipeline.classes_[predicted_idx]
        
        # Get confidence (max probability)
        confidence = float(np.max(probabilities))
        
        return predicted_category, confidence, probabilities
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict categories for multiple ticket texts.
        
        Args:
            texts: List of ticket text strings
            
        Returns:
            List of tuples (predicted_category, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        results = []
        for text in texts:
            category, confidence, _ = self.predict(text)
            results.append((category, confidence))
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model file
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    def load(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        self.is_trained = True


def train_model_from_csv(csv_path: str) -> TicketClassifier:
    """
    Train a TicketClassifier model from a CSV file.
    
    CSV should have columns: 'text' and 'label'
    
    Args:
        csv_path: Path to CSV file with training data
        
    Returns:
        Trained TicketClassifier instance
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    # Prepare data
    X = df['text'].tolist()
    y = df['label'].tolist()
    
    # Validate labels
    for label in y:
        if label not in CATEGORIES:
            raise ValueError(f"Invalid label '{label}'. Must be one of {CATEGORIES}")
    
    # Create and train classifier
    classifier = TicketClassifier()
    classifier.train(X, y)
    
    return classifier

