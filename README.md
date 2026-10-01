# Fake News Detection — NLP & Machine Learning

An end-to-end Natural Language Processing and Machine Learning project for classifying news articles into the `Fake` and `True` classes provided by the dataset.

The project covers the complete workflow from data understanding and quality checks to exploratory text analysis, TF-IDF feature extraction, model comparison, cross-validation, hyperparameter tuning, evaluation, error analysis, model interpretation, reusable prediction, and automated testing.

The final solution uses TF-IDF based text features with a tuned Linear Support Vector Machine (`LinearSVC`) classifier.

---

## Project Overview

The goal of this project was to build a structured NLP and Machine Learning workflow for news text classification rather than only training a single model.

The project was developed through four notebooks, with reusable Python scripts created after the experimentation workflow.

The complete work includes:

- Data loading and understanding
- Data quality and consistency checks
- Missing and empty value analysis
- Duplicate and conflicting-label analysis
- Exploratory text analysis
- Text statistics and vocabulary analysis
- Unigram and bigram analysis
- TF-IDF feature extraction
- Machine learning model comparison
- Stratified cross-validation
- Hyperparameter tuning using GridSearchCV
- Final model evaluation
- Confusion matrix and ROC-AUC analysis
- Validation error analysis
- Uncertain prediction analysis
- Model feature interpretation
- Reusable prediction functions
- Saved machine learning model
- Automated testing
- Reproducible project structure

The final trained model is saved as a complete scikit-learn pipeline containing both the TF-IDF vectorizer and LinearSVC classifier.

---

## Project Goals

The main goals of this project were to:

- Understand the structure and quality of the news dataset.
- Perform data quality checks before model development.
- Identify missing, empty, duplicate, and conflicting records.
- Explore textual differences between the `Fake` and `True` classes.
- Analyse article length, title length, vocabulary, frequent terms, and n-grams.
- Build an NLP pipeline for converting news text into numerical features.
- Use TF-IDF with unigram and bigram features.
- Compare different classical machine learning algorithms.
- Prevent data leakage during model development and evaluation.
- Use stratified 5-fold cross-validation.
- Tune promising models using GridSearchCV.
- Select the final model using cross-validation performance.
- Evaluate the final model using multiple classification metrics.
- Review validation errors and uncertain predictions.
- Interpret the text features learned by the final linear model.
- Save the trained model for reuse.
- Create reusable preprocessing, training, and prediction scripts.
- Add automated tests for important project components.

---

## Dataset

The project uses the Fake and Real News Dataset, commonly distributed as the ISOT Fake News Dataset.

The original dataset contains two separate files:

- `Fake.csv`
- `True.csv`

Each file contains news articles with fields including:

- `title`
- `text`
- `subject`
- `date`

The dataset labels are used as the target for this classification task:

- `Fake` → `0`
- `True` → `1`

The primary modelling text is created by combining the article title and article body.

The `subject` and `date` fields are retained for analysis but are not used as predictive features in the final text classification workflow.

The labels are provided by the dataset. This project does not independently verify the factual accuracy of individual news articles.

### Dataset Source

The dataset was obtained from the commonly available ISOT Fake News Dataset containing `Fake.csv` and `True.csv`.

The original files are stored locally under:

```text
data/
└── raw/
    ├── Fake.csv
    └── True.csv
```

The raw dataset files are excluded from Git tracking.

---

## Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Quality Checks
   ↓
Missing / Empty Value Analysis
   ↓
Duplicate & Conflicting Label Analysis
   ↓
Exploratory Text Analysis
   ↓
Text Preparation
   ↓
Train / Validation Split
   ↓
TF-IDF Feature Extraction
   ↓
Baseline Model Comparison
   ↓
Stratified 5-Fold Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Model Selection
   ↓
Validation Evaluation
   ↓
Error Analysis
   ↓
Model Interpretation
   ↓
Model Persistence
   ↓
Reusable Prediction
   ↓
Automated Testing
```

The project is divided into four notebooks:

```text
01_data_understanding_and_quality.ipynb
        ↓
02_eda_and_text_analysis.ipynb
        ↓
03_model_development_and_evaluation.ipynb
        ↓
04_model_interpretation_and_prediction.ipynb
```

The notebooks contain the analysis and experimentation workflow, while the `src` directory contains reusable Python code.

---

## Notebook 1 — Data Understanding & Quality

The first notebook focuses on understanding the raw dataset and preparing reliable data for the modelling workflow.

### Main Work

- Loaded `Fake.csv` and `True.csv`.
- Standardised the dataset columns.
- Assigned `Fake` and `True` labels.
- Combined both datasets.
- Checked dataset dimensions and data types.
- Checked missing values.
- Checked empty titles and article text.
- Analysed duplicate records.
- Checked normalised duplicate content.
- Checked content associated with conflicting labels.
- Created the combined `model_text` field.
- Removed unsuitable records identified during quality checks.
- Validated the final dataset structure.
- Saved the cleaned modelling dataset.

The final modelling dataset contains:

```text
title
text
subject
date
label
label_name
model_text
```

The processed dataset is saved as:

```text
data/processed/news_modeling_data.csv
```

The notebook also generates the initial data-quality reports and visualisations required for later analysis.

---

## Notebook 2 — Exploratory Data Analysis & Text Analysis

The second notebook focuses on understanding the textual characteristics of the dataset before model development.

### Analysis Performed

- Class distribution analysis
- Article text length analysis
- Title length analysis
- Word-count analysis
- Uppercase character ratio
- Punctuation patterns
- Digit ratios
- Vocabulary size
- Frequent unigram analysis
- Stopword-aware content term analysis
- Class-specific vocabulary
- Frequent bigrams
- Class-specific bigrams
- Subject distribution
- Subject versus class analysis
- Text availability analysis

Text normalisation was used for exploratory frequency analysis so that common variations in casing and punctuation did not unnecessarily fragment the vocabulary.

The notebook generates reports for frequent terms and bigrams and saves visualisations under:

```text
reports/
reports/figures/
```

The exploratory CountVectorizer objects were used only for analysis and were not reused for final model training.

The final TF-IDF vectorizer was fitted separately during model development after the train/validation split to avoid validation-data leakage.

---

## Notebook 3 — Model Development & Evaluation

The third notebook contains the main machine learning workflow.

The modelling text is:

```text
title + text
```

The target is the dataset-provided binary label:

```text
Fake → 0
True → 1
```

### Train / Validation Split

The dataset was divided using a stratified train/validation split so that the class distribution remained consistent between the two sets.

The main split used:

```text
80% Training
20% Validation
```

### TF-IDF Configuration

The final TF-IDF configuration uses:

```python
TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000,
)
```

This converts the news text into numerical features using both individual words and two-word combinations.

### Baseline Models

Three classical machine learning models were evaluated:

- Logistic Regression
- Linear Support Vector Machine
- Multinomial Naive Bayes

The models were compared using the same text representation and evaluation workflow.

### Cross-Validation

A stratified 5-fold cross-validation strategy was used with:

```text
n_splits = 5
shuffle = True
random_state = 42
```

The cross-validation process evaluated:

- Accuracy
- Precision
- Recall
- F1-score

F1-score was used as the main metric for model comparison and selection.

### Hyperparameter Tuning

GridSearchCV was used to tune the strongest candidate models.

The tuning process explored:

```text
TF-IDF ngram_range:
(1,1)
(1,2)

TF-IDF min_df:
2
5

Model C:
0.5
1
2

class_weight:
None
balanced
```

For the LinearSVC tuning:

```text
24 parameter combinations × 5 folds
= 120 fits
```

The tuned LinearSVC achieved the strongest cross-validation F1-score among the final candidates evaluated.

### Final Evaluation

The selected model was evaluated on the held-out validation set using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

Validation predictions were also saved for detailed error analysis.

---

## Notebook 4 — Model Interpretation & Prediction

The fourth notebook focuses on understanding the final model and creating a reusable prediction workflow.

### Model Interpretation

The final LinearSVC model was inspected using its learned coefficients and TF-IDF feature names.

The project extracts the model coefficients to identify words and phrases that have stronger associations with each class within the learned dataset patterns.

The feature coefficients are saved as:

```text
reports/model_feature_coefficients.csv
```

These features represent statistical patterns learned from the dataset and are not independent evidence that an article is factually true or false.

### Reproducibility Check

The same train/validation split was recreated using the same random state to verify that the evaluation workflow remained consistent.

### Error Analysis

The notebook reviews:

- Validation errors
- Incorrect predictions
- Uncertain predictions
- Validation prediction patterns
- Subject-level validation performance

A subject-level validation analysis is also generated to examine how model performance varies across the dataset's subject categories.

### Reusable Prediction

A reusable prediction function was created so that a new article can be passed to the saved model without rerunning the complete notebook workflow.

The final prediction interface accepts:

```text
title
text
```

and returns:

```text
prediction
label
decision_score
```

---

## NLP & Feature Engineering

The main NLP feature engineering approach is based on TF-IDF.

The article title and body are combined into a single modelling text:

```text
model_text = title + text
```

The final TF-IDF representation uses:

- Lowercase conversion
- Unicode accent stripping
- English stop-word removal
- Unigrams
- Bigrams
- `min_df=2`
- `max_df=0.95`
- Sublinear TF scaling
- Maximum 100,000 features

The TF-IDF transformation is part of the final scikit-learn pipeline.

This keeps feature extraction and classification together and ensures that the vectorizer is fitted only on the appropriate training data during evaluation.

No `subject` or `date` information is used as a predictive feature in the final text classifier.

---

## Final Model

The final selected model is a tuned **Linear Support Vector Machine using `LinearSVC`** with TF-IDF text features.

The complete pipeline contains:

```text
TF-IDF Vectorizer
        ↓
LinearSVC Classifier
```

### Final LinearSVC Configuration

```text
C = 2.0
class_weight = balanced
loss = squared_hinge
penalty = l2
multi_class = ovr
random_state = 42
```

### Final TF-IDF Configuration

```text
lowercase = True
strip_accents = unicode
stop_words = english
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
max_features = 100000
```

The final trained model is saved as:

```text
models/fake_news_final_model.joblib
```

The saved file contains the fitted TF-IDF vectorizer and LinearSVC classifier as one reusable scikit-learn pipeline.

---

## Model Performance

The final model achieved the following results.

### Cross-Validation Performance

| Metric | Score |
|---|---:|
| Mean CV F1-score | 0.9961 |

### Validation Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.9964 |
| Precision | 0.9951 |
| Recall | 0.9983 |
| F1-score | 0.9967 |
| ROC-AUC | 0.9998 |

The validation results are based on the held-out validation set used during the project workflow.

These results are specific to this dataset and evaluation setup and should not be interpreted as evidence that the model can independently verify arbitrary news from the internet.

---

## Evaluation & Error Analysis

The final model was evaluated using multiple metrics instead of relying only on accuracy.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report
- Validation predictions
- Validation error analysis
- Uncertain prediction analysis
- Subject-level validation analysis

The validation predictions are stored in:

```text
reports/validation_predictions.csv
```

The final evaluation report is stored in:

```text
reports/final_model_evaluation.csv
```

The classification report is stored in:

```text
reports/classification_report.csv
```

The confusion matrix and ROC curve are saved under:

```text
reports/figures/
```

---

## Model Interpretation

Because the final classifier is a linear SVM, its learned coefficients can be inspected to understand which TF-IDF features are more strongly associated with each class.

The project extracts the learned TF-IDF feature names and classifier coefficients and stores them in:

```text
reports/model_feature_coefficients.csv
```

This provides a useful view of the textual patterns learned by the model.

The extracted terms should be understood as dataset-specific statistical patterns rather than factual evidence about the claims contained in an article.

---

## Reusable Prediction

The project includes reusable prediction code in:

```text
src/predict.py
```

A single news article can be classified using:

```python
from src.predict import predict_article

result = predict_article(
    title="Example news headline",
    text="Example article text."
)

print(result)
```

The prediction function returns:

```text
prediction
label
decision_score
```

For `LinearSVC`, the `decision_score` comes from the classifier's decision function.

It is **not a probability** and should not be interpreted as a percentage likelihood that an article is fake or true.

The prediction script can also be executed directly:

```bash
python src/predict.py
```

---

## Project Structure

```text
Fake-News-Detection/
│
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   └── processed/
│       └── news_modeling_data.csv
│
├── models/
│   └── fake_news_final_model.joblib
│
├── notebooks/
│   ├── 01_data_understanding_and_quality.ipynb
│   ├── 02_eda_and_text_analysis.ipynb
│   ├── 03_model_development_and_evaluation.ipynb
│   └── 04_model_interpretation_and_prediction.ipynb
│
├── reports/
│   ├── figures/
│   ├── baseline_model_comparison.csv
│   ├── classification_report.csv
│   ├── cross_validation_results.csv
│   ├── fake_class_bigrams.csv
│   ├── fake_class_terms.csv
│   ├── final_model_evaluation.csv
│   ├── final_model_summary.csv
│   ├── model_feature_coefficients.csv
│   ├── prediction_review.csv
│   ├── sample_predictions.csv
│   ├── top_bigrams.csv
│   ├── top_content_terms.csv
│   ├── true_class_bigrams.csv
│   ├── true_class_terms.csv
│   ├── tuned_model_comparison.csv
│   └── validation_predictions.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The project separates exploratory notebooks, reusable Python code, model files, reports, visualisations, and tests so that the workflow remains organised and reproducible.

---

## Installation

Clone the repository and move into the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The project was developed using Python 3.13.

---

## Running the Prediction Script

After activating the virtual environment, run:

```bash
python src/predict.py
```

The script loads the saved TF-IDF and LinearSVC pipeline and performs a sample prediction.

The saved model is already available under:

```text
models/fake_news_final_model.joblib
```

so the prediction script does not require model retraining.

---

## Training

The reusable training script is available at:

```text
src/train.py
```

The script builds the final TF-IDF and LinearSVC pipeline using the selected configuration and trains it on the processed modelling dataset.

Run it from the project root with:

```bash
python src/train.py
```

The trained pipeline is saved to:

```text
models/fake_news_final_model.joblib
```

The notebooks contain the complete model comparison, cross-validation, tuning, and evaluation workflow used to develop and select the final model.

---

## Automated Testing

The project includes automated tests using `pytest`.

Run the complete test suite from the project root:

```bash
python -m pytest tests -v
```

The tests verify:

- Processed dataset availability
- Required dataset columns
- Processed data loading
- Text and target preparation
- Non-empty modelling text
- Final model availability
- Final model loading
- Prediction interface
- Model prediction behaviour

Current test result:

```text
8 tests passed
```

The tests provide an automated check that the main preprocessing and prediction components continue to work correctly.

---

## Reports & Visualisations

The project generates reports and visualisations throughout the complete workflow.

### Data Quality

- Data quality report
- Class distribution
- Text length analysis
- Subject distribution

### Exploratory Text Analysis

- Word and text statistics
- Title length analysis
- Frequent content terms
- Frequent bigrams
- Fake-class terms
- True-class terms
- Fake-class bigrams
- True-class bigrams

### Model Development

- Baseline model comparison
- Cross-validation results
- Tuned model comparison
- Final model evaluation
- Classification report
- Validation predictions

### Model Evaluation & Interpretation

- Confusion matrix
- ROC curve
- Model feature coefficients
- Prediction review
- Sample predictions
- Validation accuracy by subject

Reports are stored under:

```text
reports/
```

Visualisations are stored under:

```text
reports/figures/
```

---

## Reproducibility

The project uses fixed random seeds where applicable, including `random_state=42` in the main modelling workflow.

The train/validation split and stratified cross-validation setup use fixed random states so that the evaluation workflow can be reproduced.

The final text classification system is implemented as a scikit-learn pipeline containing the TF-IDF transformation and LinearSVC classifier.

The trained pipeline is saved using Joblib.

Project dependencies are documented in:

```text
requirements.txt
```

The four notebooks preserve the complete analysis and experimentation workflow used to develop the final model.

---

## Limitations

### Dataset Dependence

The model learns patterns from the specific dataset used for training.

Strong performance on this dataset does not guarantee the same performance on news from different sources, topics, time periods, writing styles, or domains.

### Dataset Labels

The `Fake` and `True` labels are provided by the dataset.

The model therefore learns patterns associated with those dataset labels rather than independently verifying the factual accuracy of every article.

### Distribution Shift

News language, topics, writing styles, and sources can change over time.

A model trained on historical data may therefore behave differently when applied to newer or substantially different content.

### Language and Domain

The current workflow is designed around English-language news articles.

Performance may not transfer directly to other languages or substantially different domains without additional data and evaluation.

### Model Confidence

LinearSVC provides a decision function rather than calibrated probabilities.

The `decision_score` returned by the model should therefore not be interpreted as a percentage probability.

---

## Responsible Use

This project is intended as an educational and portfolio demonstration of NLP and Machine Learning techniques.

A model prediction should not be treated as a definitive factual judgement about a news article.

A real-world fact-checking system would require additional evidence beyond text classification, including factors such as:

- Source verification
- Claim-level analysis
- Evidence retrieval
- External references
- Human review
- Continuously updated data

The purpose of this project is to demonstrate how machine learning can learn classification patterns from labelled text data, not to replace professional fact-checking.

---

## Key Learning Outcomes

This project helped strengthen practical understanding of:

- Natural Language Processing
- Text classification
- TF-IDF feature extraction
- Unigrams and bigrams
- Text preprocessing
- Classical machine learning
- Logistic Regression
- Linear Support Vector Machines
- Multinomial Naive Bayes
- Stratified cross-validation
- Hyperparameter tuning
- GridSearchCV
- Model evaluation
- Confusion matrices
- ROC-AUC
- Error analysis
- Model interpretation
- Data leakage prevention
- Scikit-learn pipelines
- Model persistence with Joblib
- Reusable Python code
- Automated testing with Pytest
- Project organisation
- Reproducible machine learning workflows

---

## Final Takeaway

This project focused on building a complete and structured NLP classification workflow rather than only training a model and reporting its accuracy.

The work started with data understanding and quality checks, followed by exploratory text analysis and NLP feature engineering.

Multiple machine learning models were then compared using a consistent evaluation workflow. Stratified cross-validation and GridSearchCV were used to evaluate and tune promising models before selecting the final LinearSVC model.

The final stage included validation, error analysis, model interpretation, reusable prediction code, model persistence, and automated testing.

The project provided practical experience in taking a text-based machine learning problem from raw labelled data to a tested and reusable NLP model.

---

## Technologies Used

**Programming:** Python

**Data Analysis:** Pandas, NumPy

**Visualisation:** Matplotlib, Seaborn

**Natural Language Processing:** TF-IDF, Unigrams, Bigrams

**Machine Learning:** Scikit-learn

**Models:** Logistic Regression, LinearSVC, Multinomial Naive Bayes

**Model Selection:** Stratified K-Fold Cross-Validation, GridSearchCV

**Model Persistence:** Joblib

**Testing:** Pytest

**Development:** Jupyter Notebook, VS Code

**Version Control:** Git, GitHub

---

## Author

**Sanskriti Singh**

B.Tech Computer Science and Engineering — Artificial Intelligence

Interested in Machine Learning, Artificial Intelligence, Natural Language Processing, Data Analysis, and practical software development.

GitHub: `sanskriti-singh-AI`
