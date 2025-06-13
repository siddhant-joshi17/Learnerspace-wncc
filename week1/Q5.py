import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
feedback = ["Good product" for _ in range(50)] + ["Bad product" for _ in range(50)]
labels = ["good"] * 50 + ["bad"] * 50

feedback_df = pd.DataFrame({"Feedback": feedback, "Label": labels})

vectorizer_tfidf = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')
X2 = vectorizer_tfidf.fit_transform(feedback_df['Feedback'])
y2 = feedback_df['Label']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=42)
model_lr = LogisticRegression()
model_lr.fit(X2_train, y2_train)
y2_pred = model_lr.predict(X2_test)

print("\nLogistic Regression Metrics:")
print("Precision:", precision_score(y2_test, y2_pred, pos_label='good'))
print("Recall:", recall_score(y2_test, y2_pred, pos_label='good'))
print("F1-Score:", f1_score(y2_test, y2_pred, pos_label='good'))

def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)

print("Vectorized sample:", text_preprocess_vectorize(["Good quality"], vectorizer_tfidf).toarray())

