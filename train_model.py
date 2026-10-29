
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.pipeline import Pipeline



# LOAD THE DATASET


print("Loading dataset...")

df = pd.read_csv("data/fake_job_postings.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)



# PREPARE THE TEXT DATA


text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Replace missing text values with empty strings
df[text_columns] = df[text_columns].fillna("")

# Combine the selected text columns into one column
df["combined_text"] = df[text_columns].agg(" ".join, axis=1)

# Separate input text and target labels
X = df["combined_text"]
y = df["fraudulent"]

print("\nText preparation completed.")



# SPLIT THE DATASET


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset split completed.")
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))



# cREATE THE MACHINE LEARNING PIPELINE


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])



# TRAIN THE MODEL


print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed.")



# EVALUATE THE MODEL


print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))



# SAVE THE TRAINED MODEL


joblib.dump(model, "fake_job_detector.joblib")

print("\nModel saved as fake_job_detector.joblib")

print("\nAll steps completed successfully!")