"""Script to train and save the ticket classification model."""

from classifier.model import train_model_from_csv
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Retrain the model
print("Training model on updated dataset...")
classifier = train_model_from_csv('data/sample_tickets.csv')

# Save it for later use
model_path = 'models/trained_model.pkl'
classifier.save(model_path)
print(f"Model retrained successfully and saved to {model_path}!")

