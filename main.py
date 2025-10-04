import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("email.csv") 

# Split into features and target
X = df['email_text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text → TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)#learn vocabulary + convert text to numbers
X_test_vec = vectorizer.transform(X_test)#only convert text to numbers. use learned vocabulary from training  data

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
'''print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))'''

new_email = ["We are delighted to offer you a position at our company!"]
new_email_vec = vectorizer.transform(new_email)
prediction = model.predict(new_email_vec)

if prediction[0] == 1:
    print("Acceptance Email")
else:
    print("Rejection Email")
