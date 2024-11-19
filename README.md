# Oasis-TASK4-SPAM

# Email Spam Detection with Machine Learning

## Overview
This project aims to build an **Email Spam Detector** using Python and machine learning techniques. The detector classifies emails as **spam** or **ham (non-spam)** by analyzing their textual content.

---

## Key Steps

### 1. **Dataset**
- **Source**: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description**: Contains 5,572 labeled SMS messages:
  - `ham`: legitimate messages
  - `spam`: unwanted or fraudulent messages

### 2. **Preprocessing**
- Imported necessary libraries like `pandas`, `numpy`, and `sklearn`.
- Downloaded the dataset using the `opendatasets` library.
- **Raw Data**: The dataset initially had 5 columns. Only two columns (`v1` and `v2`) were retained:
  - `v1`: Labels (`ham` or `spam`)
  - `v2`: Message content
- **Label Encoding**:
  - `ham` → `1`
  - `spam` → `0`
- Dropped unnecessary columns and checked dataset shape: **5,572 rows, 2 columns**.

### 3. **Data Splitting**
- Split data into training and testing sets:
  - **Training Data**: 4,457 samples
  - **Testing Data**: 1,115 samples
- Used `train_test_split` with a test size of 20% and random state of 3.

### 4. **Feature Extraction**
- Converted text data into numerical vectors using **TF-IDF Vectorization**:
  - `min_df=1`: Minimum document frequency
  - `stop_words='english'`: Removed common English stop words
  - `lowercase=True`: Converted all text to lowercase

### 5. **Model Building**
- Used **Logistic Regression** from `sklearn`:
  - Trained the model on vectorized training data.
  - Predicted results on both training and test data.

### 6. **Performance Evaluation**
- **Training Accuracy**: 96.61%
- **Testing Accuracy**: 96.23%
- The model performs well with a high accuracy in detecting spam.

### 7. **Prediction Example**
- Tested the model with a sample input:
  - **Message**: `"Hello! How's you and how did saturday go? I was just texting to see if you'd decided to do anything tomo. Not that i'm trying to invite myself or anything!"`
  - **Prediction**: `Ham` (non-spam)

---

## Code Summary
- **Libraries Used**: 
  - Data handling: `pandas`, `numpy`
  - Vectorization: `TfidfVectorizer`
  - Model: `LogisticRegression`
  - Metrics: `accuracy_score`
- **Steps**: Data preparation → Vectorization → Model training → Evaluation → Testing.

---

## Results
- **Training Accuracy**: **96.61%**
- **Test Accuracy**: **96.23%**

---

## How to Run
1. Install dependencies:
   ```bash
   pip install opendatasets numpy pandas scikit-learn
   ```
2. Download the dataset using Kaggle credentials.
3. Run the provided Python script in a Jupyter Notebook or any Python environment.
4. Test the model with your own email content as input.

---
