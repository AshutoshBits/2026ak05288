# Machine Learning Assignment 2

## Handwritten Digit Classification

### A. Problem statement

Predict which digit from 0 to 9 is represented by an 8 × 8 grayscale image. Compare five classifiers on the same held-out data, then use a Streamlit interface to evaluate the saved models on an uploaded test CSV.

### B. Dataset description

The experiment uses the 1,797-image subset of the UCI Optical Recognition of Handwritten Digits dataset bundled with scikit-learn. Each image is represented by 64 pixel measurements with values from 0 to 16. The target column is `digit`. [UCI dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) · [scikit-learn dataset loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

A stratified 75/25 split with random seed 27 produced 1,347 training rows and 450 test rows. The same split was used for every classifier. The repository's `test_data.csv` contains the 64 features and the true digit label for each held-out row.

### C. GitHub repository link

**Repository URL** `<paste your actual GitHub repository URL here>`

### D. Models and evaluation

Accuracy is the fraction of correctly classified test images. Precision, recall and F1 are macro-averaged, so each digit receives equal weight. AUC is the macro average of one-vs-rest ROC AUC using class probabilities. MCC measures agreement between the predicted and true labels across all classes.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9800 | 0.9995 | 0.9802 | 0.9800 | 0.9800 | 0.9778 |
| Decision Tree | 0.8467 | 0.9145 | 0.8460 | 0.8460 | 0.8455 | 0.8297 |
| K-Nearest Neighbors | 0.9756 | 0.9951 | 0.9769 | 0.9755 | 0.9758 | 0.9729 |
| Gaussian Naive Bayes | 0.8556 | 0.9796 | 0.8785 | 0.8547 | 0.8567 | 0.8416 |
| Support Vector Machine | 0.9867 | 0.9978 | 0.9869 | 0.9869 | 0.9868 | 0.9852 |

#### Model observations

| Model | Observation from the test results |
| --- | --- |
| Logistic Regression | Correctly classified 441 of 450 images. Its AUC of 0.9995 was the highest, and its accuracy was 0.9800. |
| Decision Tree | Correctly classified 381 images. Its F1 of 0.8455 and MCC of 0.8297 were the lowest among the five models. |
| K-Nearest Neighbors | Correctly classified 439 images, two fewer than Logistic Regression. Its macro F1 was 0.9758. |
| Gaussian Naive Bayes | Its AUC of 0.9796 was higher than its macro F1 of 0.8567. The model ranked digits well by class probability, while its final predictions were correct for 385 images. |
| Support Vector Machine | Correctly classified 444 images, the most of the five models. It also had the highest macro F1 and MCC. |

**Overall winner** The Support Vector Machine had the highest accuracy (0.9867), macro F1 (0.9868) and MCC (0.9852). Accuracy and MCC are the criteria for this choice because the app evaluates the final digit predictions.

### Run the Streamlit app

With Python 3.12, run these commands from the project folder.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

In the app, upload `test_data.csv` and choose a classifier. The app loads the corresponding fitted model from `model/` and displays the six metrics and the confusion matrix.

### Results check

The supplied models were exported with scikit-learn 1.9.1. The values above were recalculated from the supplied model files and test CSV in a local check using scikit-learn 1.8.0. Confirm the table and observations against your BITS Virtual Lab results under 1.9.1 before submission.
