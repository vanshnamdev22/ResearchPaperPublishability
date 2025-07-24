# PDF Publishability Classifier

This project uses Natural Language Processing (NLP) and Machine Learning to automatically classify research papers (PDFs) as **Publishable** or **Non-Publishable** based on their content.

---

## 📁 Directory Structure

Update the paths in the script with your actual directory locations:

```
/Reference/
├── Publishable/         ← Folder with publishable papers (label = 1)
├── Non-Publishable/     ← Folder with non-publishable papers (label = 0)

/Papers/                 ← Folder with new papers to predict
```

---

## 🧰 Dependencies

Install the required Python packages:

```bash
pip install PyPDF2 scikit-learn pandas
```

---

## 🚀 How It Works

### 1. Extract and Clean Text
- Extracts text from PDFs using `PyPDF2`.
- Cleans text using regular expressions (removes special characters, lowercases, trims extra spaces).

### 2. Vectorize Text
- Applies **TF-IDF Vectorization** (with bigrams and max 2000 features).

### 3. Train Model
- Trains a **Random Forest Classifier**.
- Uses **GridSearchCV** to find the best hyperparameters based on F1 Score.

### 4. Evaluate Model
- Measures performance using:
  - **Accuracy**
  - **F1 Score**
  - **Classification Report**

### 5. Predict New PDFs
- Reads and classifies new PDFs from the `/Papers/` folder.
- Labels each as either:
  - `Publishable`
  - `Non-Publishable`

---

## 📈 Sample Output

```
Accuracy: 92.50%
F1 Score: 0.91

Classification Report:
              precision    recall  f1-score   support
           0       0.91      0.94      0.92        36
           1       0.94      0.90      0.92        34
    accuracy                           0.92        70
```

### Example Predictions:

```
Predictions for all papers in the folder:
paper1.pdf: Publishable
paper2.pdf: Non-Publishable
...
```

---

## 📌 Notes

- Make sure your folders contain only `.pdf` files.
- The TF-IDF model doesn’t understand context or semantics.
- For better results on complex documents, consider transformer-based models (e.g., BERT or SciBERT).
- Preprocessing assumes machine-readable PDFs (not scanned images).

---

## 📄 License

This project is open for research and educational use. You may modify or adapt it for your needs.
