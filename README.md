AI_Ass2_Code

This repository contains the Python implementation used for Assessment 2 – Machine Learning AI Scientific Research for the module CPU5006 Artificial Intelligence (Bath Spa University).

The project compares the performance of two supervised machine learning classifiers:

Decision Tree

k-Nearest Neighbours (KNN)

Both models are trained and evaluated using the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository.

📄 Associated Research Paper

Title:
Comparing Decision Tree and k-Nearest Neighbours for Predicting Breast Cancer Diagnosis Using the Wisconsin Diagnostic Dataset

This repository provides the supporting code used to generate all experimental results reported in the paper.

📊 Dataset

Source:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset contains 569 samples with 30 numerical features derived from digitised images of breast mass cell nuclei.
Target labels:

malignant

benign

🧠 Models Implemented
Decision Tree Classifier

Implemented using DecisionTreeClassifier from Scikit-learn.

Gini impurity criterion.

Fixed random seed for reproducibility.

No feature scaling required.

k-Nearest Neighbours (KNN)

Implemented using KNeighborsClassifier from Scikit-learn.

k = 5 neighbours.

Uses Euclidean distance.

All features are scaled using StandardScaler prior to training.

🛠️ Files in This Repository
File	Description
breast_Cancer_ML.py	Main Python script that loads the dataset, preprocesses features, trains both models, evaluates performance, and outputs metrics.
breast_cancer_model_comparison.csv	CSV output containing performance summaries for both classifiers (accuracy, precision, recall, F1-score).
README.md	Project documentation (this file).
.sln / project files	Visual Studio project configuration files created during development in Visual Studio 2022.
▶️ Running the Code
Requirements

Ensure Python is installed (3.9+ recommended), then install dependencies:

pip install numpy pandas scikit-learn

Run

From the repository folder:

python breast_Cancer_ML.py

✅ Output

When executed, the script will:

Load and preprocess the dataset.

Train both Decision Tree and KNN models.

Evaluate classifiers using:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Display detailed classification reports.

Export a performance comparison table to:

breast_cancer_model_comparison.csv

📈 Evaluation Metrics

Performance is evaluated on an 80/20 stratified train–test split using:

Accuracy – overall prediction correctness

Precision – reliability of malignancy predictions

Recall (Sensitivity) – ability to identify true malignant cases (clinically critical)

F1-score – balanced assessment of precision and recall

Confusion Matrix – visual breakdown of classification errors

🎯 Purpose of the Experiment

The goal of this project is to determine:

Which classification method — Decision Tree or k-Nearest Neighbours — provides more effective diagnostic performance for breast cancer prediction?

Findings indicate that:

KNN achieved higher recall and overall accuracy, producing fewer false negatives.

Decision Trees offered interpretability but lower sensitivity for malignant tumour detection.

🔁 Reproducibility

All experiments are:

Fully deterministic due to the use of fixed random seeds.

Repeatable by running the same script using the dataset specified above.

🤝 Academic Use of AI Tools

Where AI assistance (e.g., ChatGPT) was used to support code development or report preparation, outputs were reviewed, validated, and manually edited to ensure correctness and originality, in accordance with university guidance.
