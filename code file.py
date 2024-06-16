# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z12eEaXiegSBhoe1d12hraNevkbVJfWB
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/content/Monkeypox Coursework Dataset (1).csv')

retained_variables = ['Systemic Illness', 'Sore Throat', 'Rectal Pain', 'Penile Oedema', 'Oral Lesions',
                       'Solitary Lesion', 'Swollen Tonsils', 'HIV Infection',
                       'Home ownership', 'Age', 'Health Insurance', 'Sexually Transmitted Infection', 'MPOX PCR Result']

# Basic statistical description
description = df[retained_variables].describe()
print (description)

# Measurement scale type
measurement_scale = df[retained_variables].dtypes
print(measurement_scale)

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loading data into a DataFrame
df=pd.read_csv('/content/Monkeypox Coursework Dataset (1).csv')

# Plot the distribution of the class variable 'MPOX'
plt.figure(figsize=(8, 6))
sns.countplot(x='MPOX PCR Result', data=df)
plt.title('Distribution of MPOX Classes')
plt.xlabel('MPOX PCR Result')
plt.ylabel('Count')
plt.show()

df[retained_variables].info()

# Measurement scale type
measurement_scale = df[retained_variables].dtypes
print(measurement_scale)

df[retained_variables].isnull().sum()

for column in retained_variables:
    unique_values = df[column].unique()
    print(f"Unique values in {column}:", unique_values)

df.duplicated().sum()

df[retained_variables].info()

# Import necessary libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

# Check for missing values in the entire dataset
missing_values = df.isnull().sum()
print("\nMissing values in each variable:")
print(missing_values)

# Visualize distributions of numerical variables
numerical_variables = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_variables].hist(bins=20, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Variables')
plt.show()

# Detect outliers using z-scores
z_scores = np.abs(stats.zscore(df[numerical_variables]))
outliers = (z_scores > 3).all(axis=1)
print(f"\nNumber of rows with outliers: {outliers.sum()}")

# Check data types of each variable
data_types = df.dtypes
print("\nData types of each variable:")
print(data_types)

# Check for class imbalance in the target variable
class_distribution = df['MPOX PCR Result'].value_counts()
print("\nClass distribution of MPOX PCR Result:")
print(class_distribution)

# Explore correlations between variables
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#Handling Missing Values for Systemic Illness:
# Standardize representations for 'Fever'
df['Systemic Illness'] = df['Systemic Illness'].replace({'fever': 'Fever'})

# Fill missing values with the mode
df['Systemic Illness'].fillna(df['Systemic Illness'].mode()[0], inplace=True)

#Handling Missing Values for Rectal Pain:
df.dropna(subset=['Rectal Pain'], inplace=True)

#Handling Missing Values for Penile Oedema:
df['Penile Oedema'].fillna(df['Penile Oedema'].mean(), inplace=True)

#Handling Missing Values for Oral Lesions:
df['Oral Lesions'].replace({'YES': '1', 'No': '0'}, inplace=True)
df['Oral Lesions'].fillna(df['Oral Lesions'].mode()[0], inplace=True)
df['Oral Lesions'] = df['Oral Lesions'].astype(int)

#Handling Missing Values for Swollen Tonsils:
df['Swollen Tonsils'].fillna(df['Swollen Tonsils'].median(), inplace=True)

#Handling Missing Values for HIV Infection:
df.dropna(subset=['HIV Infection'], inplace=True)

#Handling Missing Values for Age:


# Convert non-numeric values to NaN and then fill with the median
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Replace negative values and unrealistic age values with the median
df['Age'] = df['Age'].apply(lambda x: median_age if x < 0 or x > 120 else x)

#Handling Missing Values for Sexually Transmitted Infection:
df.dropna(subset=['Sexually Transmitted Infection'], inplace=True)

# Convert 'MPOX PCR Result' to binary representation
df['MPOX PCR Result'] = df['MPOX PCR Result'].map({'Negative': 0, 'Positive': 1})

for column in retained_variables:
    unique_values = df[column].unique()
    print(f"Unique values in {column}:", unique_values)

# Measurement scale type
measurement_scale = df[retained_variables].dtypes
print(measurement_scale)

# Missing values after handling
print(df[retained_variables].isnull().sum())

for column in retained_variables:
    unique_values = df[column].unique()
    print(f"Unique values in {column}:", unique_values)

# Measurement scale type
measurement_scale = df[retained_variables].dtypes
print(measurement_scale)

df[retained_variables].isnull().sum()

from sklearn.model_selection import train_test_split

# Assuming 'X' is your feature matrix and 'y' is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the shapes of the training and test sets
print("Training set shape - X:", X_train.shape, " y:", y_train.shape)
print("Test set shape - X:", X_test.shape, " y:", y_test.shape)

import pandas as pd
from sklearn.model_selection import train_test_split


# Assuming 'df' is your original DataFrame
X = df[retained_variables].drop(columns=['MPOX PCR Result'])  # Features
y = df[retained_variables ]['MPOX PCR Result']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create DataFrames for training data without the target column
df_train_X = X_train.copy()
df_train_y = pd.DataFrame(y_train, columns=['MPOX PCR Result'])

# Display the shapes of the new DataFrames
print("Training data shape without target column:", df_train_X.shape)
print("Training target column shape:", df_train_y.shape)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

retained_variables = ['Systemic Illness', 'Sore Throat', 'Rectal Pain', 'Penile Oedema', 'Oral Lesions',
                       'Solitary Lesion', 'Swollen Tonsils', 'HIV Infection',
                       'Home ownership', 'Age', 'Health Insurance', 'Sexually Transmitted Infection', 'MPOX PCR Result']

X = df[retained_variables[:-1]]  # Features
y = df['MPOX PCR Result']        # Target variable

from sklearn.preprocessing import LabelEncoder

# Assuming 'X_train' contains your training data
# Assuming 'X_test' contains your test data

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply the label encoder to each column with categorical data
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        X_train[column] = label_encoder.fit_transform(X_train[column])
        X_test[column] = label_encoder.transform(X_test[column])

# Now, your categorical variables are encoded numerically
# Proceed to fit your Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Naïve Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# SVM with RBF Kernel
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have trained models named nb_model, dt_model, lr_model, svm_model

# Make predictions
nb_predictions = nb_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)

# Confusion Matrix for Naïve Bayes
cm_nb = confusion_matrix(y_test, nb_predictions)

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test, dt_predictions)

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, lr_predictions)

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, svm_predictions)

# Plot the confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Naïve Bayes
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix - Naïve Bayes')

# Decision Tree
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix - Decision Tree')

# Logistic Regression
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix - Logistic Regression')

# SVM
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix - SVM')

plt.tight_layout()
plt.show()

# Example for Logistic Regression
from sklearn.linear_model import LogisticRegression

# Instantiate the model
lr_model = LogisticRegression(random_state=42)

# Train the model
lr_model.fit(X_train, y_train)

# Predictions on the test set
nb_predictions = nb_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
conf_matrix_nb = confusion_matrix(y_test, nb_predictions)
print("Confusion Matrix - Naïve Bayes:")
print(conf_matrix_nb)

# Classification Report
class_report_nb = classification_report(y_test, nb_predictions)
print("\nClassification Report - Naïve Bayes:")
print(class_report_nb)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions on the test set
dt_predictions = dt_model.predict(X_test)

# Confusion Matrix
conf_matrix_dt = confusion_matrix(y_test, dt_predictions)
print("Confusion Matrix - Decision Tree:")
print(conf_matrix_dt)

# Classification Report
class_report_dt = classification_report(y_test, dt_predictions)
print("\nClassification Report - Decision Tree:")
print(class_report_dt)

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Predictions on the test set
lr_predictions = lr_model.predict(X_test)

# Confusion Matrix
conf_matrix_lr = confusion_matrix(y_test, lr_predictions)
print("Confusion Matrix - Logistic Regression:")
print(conf_matrix_lr)

# Classification Report
class_report_lr = classification_report(y_test, lr_predictions)
print("\nClassification Report - Logistic Regression:")
print(class_report_lr)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Predictions
knn_predictions = knn_model.predict(X_test)

# Confusion Matrix and Classification Report - K-Nearest Neighbors
conf_matrix_knn = confusion_matrix(y_test, knn_predictions)
class_report_knn = classification_report(y_test, knn_predictions, zero_division=1)

# Print Confusion Matrix
print("Confusion Matrix - K-Nearest Neighbors:")
print(conf_matrix_knn)

# Print Classification Report
print("\nClassification Report - K-Nearest Neighbors:")
print(class_report_knn)

# Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Predictions on the test set
svm_predictions = svm_model.predict(X_test)

# Confusion Matrix
conf_matrix_svm = confusion_matrix(y_test, svm_predictions)
print("Confusion Matrix - Support Vector Machine:")
print(conf_matrix_svm)

# Classification Report
class_report_svm = classification_report(y_test, svm_predictions)
print("\nClassification Report - Support Vector Machine:")
print(class_report_svm)

# Classification Report - Support Vector Machine
class_report_svm = classification_report(y_test, svm_predictions, zero_division=1)
print("\nClassification Report - Support Vector Machine:")
print(class_report_svm)

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Function to plot ROC curve
def plot_roc_curve(y_true, y_scores, label):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

# Naïve Bayes
nb_probs = nb_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, nb_probs, 'Naïve Bayes')

# Decision Tree
dt_probs = dt_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, dt_probs, 'Decision Tree')

# Logistic Regression
lr_probs = lr_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, lr_probs, 'Logistic Regression')

# Support Vector Machine
svm_probs = svm_model.decision_function(X_test)
plot_roc_curve(y_test, svm_probs, 'Support Vector Machine')

# K-Nearest Neighbors
knn_probs = knn_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, knn_probs, 'K-Nearest Neighbors')

# Plotting settings
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

# Define the model
nb_model = GaussianNB()

# Define hyperparameters and their potential values
param_grid = {
    'priors': [None, [0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Retrieve the best hyperparameters
best_params = grid_search.best_params_

# Re-train the model with the best hyperparameters
best_nb_model = GaussianNB(**best_params)
best_nb_model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix

# Predictions on the test set
y_pred_test = best_nb_model.predict(X_test)

# Confusion matrix
cm_test = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix - Naïve Bayes (Tuned):")
print(cm_test)

from sklearn.metrics import classification_report

# Classification report
cr_test = classification_report(y_test, y_pred_test)
print("Classification Report - Naïve Bayes (Tuned):")
print(cr_test)

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}
nb_model = GaussianNB()
grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_nb_model = grid_search.best_estimator_
best_nb_model.fit(X_train, y_train)
y_pred_tuned = best_nb_model.predict(X_test)
conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)
print("Confusion Matrix - Tuned Naïve Bayes:")
print(conf_matrix_tuned)
# Use the appropriate metrics functions to calculate scores
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
auc_roc_tuned = roc_auc_score(y_test, y_pred_tuned)

# Display the scores
print("Accuracy - Tuned Naïve Bayes:", accuracy_tuned)
print("Recall - Tuned Naïve Bayes:", recall_tuned)
print("Precision - Tuned Naïve Bayes:", precision_tuned)
print("F1 Score - Tuned Naïve Bayes:", f1_tuned)
print("AUC-ROC Score - Tuned Naïve Bayes:", auc_roc_tuned)

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Base Learners
nb_model = GaussianNB()
svm_model = SVC(kernel='rbf', random_state=42)

# Fit base learners on the training data
nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Ensemble Voting Classifier
ensemble_model = VotingClassifier(estimators=[('NB', nb_model), ('SVM', svm_model)], voting='hard')

# Fit the ensemble model on the training data
ensemble_model.fit(X_train, y_train)

# Predictions on the test set
y_pred_ensemble = ensemble_model.predict(X_test)

# Confusion Matrix for Naïve Bayes
conf_matrix_nb = confusion_matrix(y_test, nb_model.predict(X_test))
print("Confusion Matrix - Naïve Bayes:")
print(conf_matrix_nb)

# Confusion Matrix for SVM with RBF kernel
conf_matrix_svm = confusion_matrix(y_test, svm_model.predict(X_test))
print("\nConfusion Matrix - SVM with RBF kernel:")
print(conf_matrix_svm)

# Confusion Matrix for Ensemble Voting Learner
conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)
print("\nConfusion Matrix - Ensemble Voting Learner:")
print(conf_matrix_ensemble)

# Calculate Metrics for each model
accuracy_nb = accuracy_score(y_test, nb_model.predict(X_test))
accuracy_svm = accuracy_score(y_test, svm_model.predict(X_test))
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)

recall_nb = recall_score(y_test, nb_model.predict(X_test))
recall_svm = recall_score(y_test, svm_model.predict(X_test))
recall_ensemble = recall_score(y_test, y_pred_ensemble)