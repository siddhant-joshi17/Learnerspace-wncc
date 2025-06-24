import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import gensim.downloader as api

model = api.load("word2vec-google-news300)

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load dataset
df = pd.read_csv("Tweets.csv")[["airline_sentiment", "text"]]

w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Clean text
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s']", "", text)
    text = re.sub(r"’", "'", text)
    contractions = {"don't": "do not", "i'm": "i am", "it's": "it is"}
    for key, val in contractions.items():
        text = text.replace(key, val)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens

df['tokens'] = df['text'].apply(clean_tweet)
df['vectors'] = df['tokens'].apply(vectorize)

# Prepare data
X = np.stack(df['vectors'].values)
le = LabelEncoder()
y = le.fit_transform(df['airline_sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model2.predict(X_test)))

# Prediction function
def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = clean_tweet(tweet)
    vector = vectorize(tokens)
    return le.inverse_transform(model.predict([vector]))[0]
