import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from preprocess.clean_text import clean_text
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/sample_tickets.csv')

X = df['text'].tolist()
y = df['label'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from classifier.model import TicketClassifier
classifier = TicketClassifier()
classifier.train(X_train, y_train)

y_pred = []
for text in X_test:
    cleaned = clean_text(text)
    pred, _, _ = classifier.predict(cleaned)
    y_pred.append(pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['Billing', 'Technical', 'Account', 'Feedback']))


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'))
])

# Cross-validation
'''cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")'''