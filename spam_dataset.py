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
X_train, X_test, y_train, y_tes# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (email content and labels: 1 for spam, 0 for ham)
emails = [
    "Free vacation! Claim your prize now!",
    "Meeting Reminder for Tomorrow",
    "Limited Time Offer: Save 50% Now",
    "Lunch Plans",
    "Get Rich Quick!",
    "Project Update",
    "Win an iPhone Now!",
    "Invoice for Last Month",
    "Urgent: Your Account is Compromised!",
    "Dinner Reservations"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for ham

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Convert the email text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Predict on the test set
y_pred = model.predict(X_test_counts)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))t = train_test_split(df['text_cleaned'], df['label'], test_size=0.2, random_state=42)

# Convert text data to numerical format using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Now X_train_tfidf and y_train can be used to train your spam detection model