import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["Label", "Message"]

# Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Load Word2Vec
w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Average Word2Vec for a message
def vectorize(tokens):
    vectors = [w2v[word] for word in tokens if word in w2v]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

df['tokens'] = df['Message'].apply(preprocess_text)
df['vectors'] = df['tokens'].apply(vectorize)

# Prepare data
X = np.stack(df['vectors'].values)
y = df['Label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Prediction function
def predict_message_class(model, w2v_model, message):
    tokens = preprocess_text(message)
    vector = vectorize(tokens)
    return 'spam' if model.predict([vector])[0] else 'ham'
