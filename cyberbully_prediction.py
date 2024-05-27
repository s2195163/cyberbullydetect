# -*- coding: utf-8 -*-
"""Cyberbully Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G9W1ko821IQpH8ZTX80Q5Cs6nlneW7P7
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import re

# Specify file paths
file_path_vid1 = "/content/drive/My Drive/WQD7002/Vid1.csv"
file_path_vid2 = "/content/drive/My Drive/WQD7002/Vid2.csv"
file_path_vid3 = "/content/drive/My Drive/WQD7002/Vid3.csv"
file_path_vid4 = "/content/drive/My Drive/WQD7002/Vid4.csv"
file_path_vid5 = "/content/drive/My Drive/WQD7002/Vid5.csv"

# Load datasets into data frames
df_vid1 = pd.read_csv(file_path_vid1)
df_vid2 = pd.read_csv(file_path_vid2)
df_vid3 = pd.read_csv(file_path_vid3)
df_vid4 = pd.read_csv(file_path_vid4)
df_vid5 = pd.read_csv(file_path_vid5)

# Change column names for df_vid4
df_vid4.columns = ["No", "Unique.ID", "Name", "Date", "Likes", "Comment", "Profile.ID"]

# Change column names for df_vid5
df_vid5.columns = ["No", "Unique.ID", "Name", "Date", "Likes", "Comment", "Profile.ID"]

# Combine all datasets into one
combined_df = pd.concat([df_vid1, df_vid2, df_vid3, df_vid4, df_vid5], ignore_index=True)

# Display the structure of the data frame
print(combined_df.info())

# Display the first few rows of the data frame
print(combined_df.head())

# Check summary statistics
print(combined_df.describe)

# Check for missing values
print(combined_df.isna().sum())  # No missing values available

# Check for duplicate rows
duplicated_rows = combined_df[combined_df.duplicated()]

# Remove special characters from comments
combined_df['Comment'] = combined_df['Comment'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)

# Function to remove emojis from a string
def remove_emojis(text):
    return text.encode('ascii', 'ignore').decode('ascii')

# Apply the function to the 'Comment' and 'Name' columns
combined_df['Comment'] = combined_df['Comment'].apply(remove_emojis)
combined_df['Name'] = combined_df['Name'].apply(remove_emojis)

# Remove numbers from the 'Comment' column
combined_df['Comment'] = combined_df['Comment'].str.replace('[0-9]', '', regex=True)

# Remove punctuation from the 'Comment' column
combined_df['Comment'] = combined_df['Comment'].str.replace('[[:punct:]]', '', regex=True)

# Remove leading and trailing whitespace from the 'Comment' column
combined_df['Comment'] = combined_df['Comment'].str.strip()

# CASE FOLDING
# Lowercase
combined_df['Unique.ID'] = combined_df['Unique.ID'].str.lower()
combined_df['Name'] = combined_df['Name'].str.lower()
combined_df['Comment'] = combined_df['Comment'].str.lower()

# Remove variable that is not needed (removing column 'Name')
combined_df = combined_df.drop(columns=['Name'])

# SHORT FORMS
# Define the dictionary for short forms
dictionary = {
    "yg": "yang",
    "tu": "itu",
    "tuu": "itu",
    "ni": "ini",
    "u": "you",
    "cm": "cam",
    "x": "tidak",
    "tak": "tidak",
    "tdk": "tidak",
    "nk": "nak",
    "je": "sahaja",
    "sya": "saya",
    "sy": "saya",
    "dah": "sudah",
    "dh": "sudah",
    "suda": "sudah",
    "org": "orang",
    "jgn": "jangan",
    "tau": "tahu",
    "tauu": "tahu",
    "dr": "dari",
    "dri": "dari",
    "dgn": "dengan",
    "ngn": "dengan",
    "kt": "dekat",
    "kat": "dekat",
    "dkt": "dekat",
    "dekt": "dekat",
    "mcm": "macam",
    "cm": "macam",
    "mna": "mana",
    "bajumcm": "baju macam",
    "lagicm": "lagi macam",
    "ank": "anak",
    "anaktahniah": "anak tahniah",
    "anknya": "anak nya",
    "sdirianknya": "sendiri anak nya",
    "abadiank": "abadi anak",
    "umo": "umur",
    "tahun": "tahun",
    "aurattaklah": "aurat tidak lah",
    "tukadang": "itu kadang",
    "jugapakaikan": "juga memakaikan",
    "a n a k azhar p e l a c you r": "anak azhar pelacur",
    "wtffff": "what the fuck",
    "camsial": "macam sial"
}

# Define the function to replace short forms
def replace_short_forms_column(column, dictionary):
    for short_form, full_form in dictionary.items():
        column = re.sub(r'\b{}\b'.format(short_form), full_form, column, flags=re.IGNORECASE)
    return column

# Apply the function to the 'Comment' column
combined_df['Comment'] = combined_df['Comment'].apply(lambda x: replace_short_forms_column(x, dictionary))

!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis
combined_df['Sentiment_Score'] = combined_df['Comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Classify sentiment as positive, negative, or neutral
combined_df['Sentiment_Label'] = combined_df['Sentiment_Score'].apply(
    lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral')
)

# Get the count of each sentiment label
sentiment_counts = combined_df['Sentiment_Label'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment_Label', 'Count']

# Print the frequency of each sentiment label
print(sentiment_counts)

# Create a bar plot of sentiment results
plt.figure(figsize=(10, 6))
plt.bar(sentiment_counts['Sentiment_Label'], sentiment_counts['Count'], color=['green', 'red', 'blue'])
plt.title('Sentiment Analysis of Comments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Define a list of abusive words in Malay
abusive_dict = ["bodoh", "menganjing", "pelacur", "yahudi", "babi", "sioniz", "latolato", "mok", "sial"]

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis
combined_df['Sentiment_Score'] = combined_df['Comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Classify sentiment as positive, negative, or neutral
combined_df['Sentiment_Label'] = combined_df['Sentiment_Score'].apply(
    lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral')
)

# Filter comments for abusive words
def contains_abusive_words(comment, abusive_dict):
    words = comment.lower().split()
    return any(word in abusive_dict for word in words)

combined_df['Contains_Abusive'] = combined_df['Comment'].apply(lambda x: contains_abusive_words(x, abusive_dict))

# Label sentiment based on abusive content
combined_df['Sentiment_Label'] = combined_df.apply(
    lambda row: 'Negative' if row['Contains_Abusive'] else row['Sentiment_Label'], axis=1
)

# Update sentiment counts
sentiment_counts = combined_df['Sentiment_Label'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment_Label', 'Count']

# Print the updated frequency of each sentiment label
print(sentiment_counts)

# Create a bar plot of updated sentiment results
plt.figure(figsize=(10, 6))
plt.bar(sentiment_counts['Sentiment_Label'], sentiment_counts['Count'], color=['green', 'red', 'blue'])
plt.title('Sentiment Analysis of Comments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'Comment' column
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['Comment'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Add the TF-IDF features to the combined_df DataFrame
combined_df = pd.concat([combined_df.reset_index(drop=True), tfidf_df], axis=1)

"""# Modeling"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

"""###Random Forest"""

# Convert Sentiment_Label to categorical if it's not already
combined_df['Sentiment_Label'] = pd.Categorical(combined_df['Sentiment_Label'])

# Split the data into training and testing sets (e.g., 70% training, 30% testing)
train_data, test_data = train_test_split(combined_df, test_size=0.3, random_state=123)

# Define feature columns (all TF-IDF features and any other relevant features except 'Sentiment_Label' and 'Contains_Abusive')
feature_columns = [col for col in combined_df.columns if col not in ['Sentiment_Label', 'Contains_Abusive', 'Comment', 'Unique.ID', 'Date', 'Profile.ID', 'No', 'Likes']]

# Split the data into training and testing sets
X_train = train_data[feature_columns]
X_test = test_data[feature_columns]
y_train = train_data['Sentiment_Label']
y_test = test_data['Sentiment_Label']

# Fit Random Forest model on the training data
rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(X_train, y_train)

# Make predictions on the testing data
predictions_rf = rf_model.predict(X_test)

# Evaluate accuracy on the testing data
accuracy_rf = accuracy_score(y_test, predictions_rf)

# Calculate additional evaluation metrics
conf_matrix_rf = confusion_matrix(y_test, predictions_rf)
precision_rf = precision_score(y_test, predictions_rf, average='macro')
recall_rf = recall_score(y_test, predictions_rf, average='macro')
f1_score_rf = f1_score(y_test, predictions_rf, average='macro')

# Calculate AUC-ROC
roc_auc_rf = roc_auc_score(pd.get_dummies(y_test), rf_model.predict_proba(X_test), multi_class='ovr')

# Create a DataFrame to display the results
results_rf_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
    "Score": [accuracy_rf, precision_rf, recall_rf, f1_score_rf, roc_auc_rf]
})

print("Random Forest Evaluation Metrics:")
print(results_rf_df)

"""### Naive Bayes"""

# Convert categorical features to numeric using one-hot encoding
train_data_encoded = pd.get_dummies(train_data[feature_columns])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data_encoded, train_data['Sentiment_Label'], test_size=0.2, random_state=123)

# Fit Naive Bayes model on the training data
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

# Make predictions on the testing data
predictions_nb = model_nb.predict(X_test)

# Evaluate accuracy on the testing data
accuracy_nb = accuracy_score(y_test, predictions_nb)

# Calculate additional evaluation metrics
conf_matrix_nb = confusion_matrix(y_test, predictions_nb)
precision_nb = precision_score(y_test, predictions_nb, average='macro')
recall_nb = recall_score(y_test, predictions_nb, average='macro')
f1_score_nb = f1_score(y_test, predictions_nb, average='macro')

# Calculate AUC-ROC
roc_auc_nb = roc_auc_score(pd.get_dummies(y_test), model_nb.predict_proba(X_test), multi_class='ovr')

# Create a DataFrame to display the results
results_nb_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
    "Score": [accuracy_nb, precision_nb, recall_nb, f1_score_nb, roc_auc_nb]
})

print("Naive Bayes Evaluation Metrics:")
print(results_nb_df)

"""### Support Vector Machine"""

from sklearn.svm import SVC

# Fit SVM model on the training data
svm_model = SVC(probability=True, random_state=123)
svm_model.fit(X_train, y_train)

# Make predictions on the testing data
predictions_svm = svm_model.predict(X_test)

# Evaluate accuracy on the testing data
accuracy_svm = accuracy_score(y_test, predictions_svm)

# Calculate additional evaluation metrics
conf_matrix_svm = confusion_matrix(y_test, predictions_svm)
precision_svm = precision_score(y_test, predictions_svm, average='macro')
recall_svm = recall_score(y_test, predictions_svm, average='macro')
f1_score_svm = f1_score(y_test, predictions_svm, average='macro')

# Calculate AUC-ROC
roc_auc_svm = roc_auc_score(pd.get_dummies(y_test), svm_model.predict_proba(X_test), multi_class='ovr')

# Create a DataFrame to display the results
results_svm_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"],
    "Score": [accuracy_svm, precision_svm, recall_svm, f1_score_svm, roc_auc_svm]
})

print("SVM Evaluation Metrics:")
print(results_svm_df)
