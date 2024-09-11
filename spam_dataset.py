import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load dataset (e.g., CSV with 'text' and 'label' columns)
df = pd.read_csv('spam_dataset.csv')

# Preprocessing: Clean the text data (example with lowercase)
df['text_cleaned'] = df['text'].str.lower()

# Encoding the labels (spam=1, ham=0)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text_cleaned'], df['label'], test_size=0.2, random_state=42)

# Convert text data to numerical format using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Now X_train_tfidf and y_train can be used to train your spam detection model