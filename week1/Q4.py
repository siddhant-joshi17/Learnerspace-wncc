import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
positive_reviews = ["I loved the movie!"] * 50
negative_reviews = ["I hated the movie."] * 50
reviews = positive_reviews + negative_reviews
sentiments = ["positive"] * 50 + ["negative"] * 50

reviews_df = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})

vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(reviews_df['Review'])
y = reviews_df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
y_pred = model_nb.predict(X_test)
print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred))

def predict_review_sentiment(model, vectorizer, review):
    features = vectorizer.transform([review])
    return model.predict(features)[0]

print("Sample Prediction:", predict_review_sentiment(model_nb, vectorizer, "Great movie!"))
