import os
!pip install PyPDF2
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import re

# Function to clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower()  # Convert to lowercase

# Function to extract text from PDFs in a given directory
def extract_text_from_pdfs(reference, label):
    data = []
    for root, _, files in os.walk(reference):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                data.append({"text": clean_text(text), "label": label})  # Clean text before appending
    return data

# Path to directories (update these paths with actual directories)
non_publishable_dir = "/content/drive/MyDrive/Reference-20250106T125528Z-001/Reference/Non-Publishable"
publishable_dir = "/content/drive/MyDrive/Reference-20250106T125528Z-001/Reference/Publishable"

# Extract text from both datasets
non_publishable_data = extract_text_from_pdfs(non_publishable_dir, label=0)
publishable_data = extract_text_from_pdfs(publishable_dir, label=1)

# Combine datasets into a single DataFrame
all_data = pd.DataFrame(non_publishable_data + publishable_data)

# Step 1: Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))  # Use bigrams
X = tfidf_vectorizer.fit_transform(all_data['text'])
y = all_data['label']

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest Model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Step 4: Evaluate the Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return clean_text(text)  # Clean text before returning

# Function to process all PDFs in a folder
def process_folder_of_pdfs(folder_path, model, vectorizer):
    results = []  # To store predictions
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):  # Check if the file is a PDF
            pdf_path = os.path.join(folder_path, file_name)
            
            try:
                # Extract text from the PDF
                paper_text = extract_text_from_pdf(pdf_path)
                
                # Vectorize the extracted text
                paper_vectorized = vectorizer.transform([paper_text])
                
                # Make prediction
                prediction = model.predict(paper_vectorized)
                prediction_label = "Publishable" if prediction[0] == 1 else "Non-Publishable"
                
                # Save the result
                results.append((file_name, prediction_label))
            except Exception as e:
                # Handle errors (e.g., unreadable PDFs)
                print(f"Error processing {pdf_path}: {e}")
    
    results.sort(key=lambda x: x[0])
    return results

# Specify the folder containing PDFs
folder_path = "/content/drive/MyDrive/Papers"  # Replace with the folder path containing PDFs

# Process the folder and generate predictions
predictions = process_folder_of_pdfs(folder_path, best_model, tfidf_vectorizer)

# Print the results
print("\nPredictions for all papers in the folder:")
for file_name, prediction_label in predictions:
    print(f"{file_name}: {prediction_label}")
