import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# 2. Encode labels: spam=1, ham=0
le = LabelEncoder()
df["label_num"] = le.fit_transform(df["label"])  # spam=1, ham=0

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42
)

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Predictions
y_pred = model.predict(X_test_tfidf)

# 7. Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Test Prediction Function
def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"

# 9. Demo predictions
print("\n🔍 Test Prediction Examples:")
print("1.", predict_spam("Congratulations! You’ve won a free ticket to Bahamas. Call now!"))
print("2.", predict_spam("Hey, are we still on for the meeting at 3pm?"))
