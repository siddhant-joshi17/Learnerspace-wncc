import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Corpus
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

# Tokenize each document
tokenized_docs = [doc.lower().split() for doc in corpus]

# Get unique words (vocabulary)
vocab = sorted(set(word for doc in tokenized_docs for word in doc))

# Term Frequency (TF)
tf = []
for doc in tokenized_docs:
    tf_doc = {}
    for word in vocab:
        tf_doc[word] = doc.count(word) / len(doc)
    tf.append(tf_doc)

# Inverse Document Frequency (IDF)
N = len(corpus)
idf = {}
for word in vocab:
    df = sum(1 for doc in tokenized_docs if word in doc)
    idf[word] = math.log(N / (df)) if df else 0

# TF-IDF
tfidf_manual = []
for doc_tf in tf:
    tfidf_doc = {}
    for word in vocab:
        tfidf_doc[word] = doc_tf[word] * idf[word]
    tfidf_manual.append(tfidf_doc)

# Convert manual result to DataFrame
manual_df = pd.DataFrame(tfidf_manual)
manual_df.index = [f"Doc{i+1}" for i in range(N)]

print("\nManual TF-IDF:")
print(manual_df.round(3))

# Comparison with Scikit-learn

# CountVectorizer
cv = CountVectorizer()
cv_matrix = cv.fit_transform(corpus)
cv_df = pd.DataFrame(cv_matrix.toarray(), columns=cv.get_feature_names_out(), index=[f"Doc{i+1}" for i in range(N)])

print("\nCountVectorizer:")
print(cv_df)

# TfidfVectorizer
tv = TfidfVectorizer()
tv_matrix = tv.fit_transform(corpus)
tv_df = pd.DataFrame(tv_matrix.toarray(), columns=tv.get_feature_names_out(), index=[f"Doc{i+1}" for i in range(N)])

print("\nScikit-learn TF-IDF:")
print(tv_df.round(3))
