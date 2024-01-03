# _**Sleep Health Lifestyle Analyzer**_

### Authored By
**Hakam Nabulssi_207710443** & **Hanna Bajjaly_211958970**

### Description

The SleepHealthLifestyleAnalyzer class provides functionality to analyze sleep health and lifestyle data using various machine learning models. It allows loading and manipulating data, exploring data, fitting different models, and evaluating their performance.

### Required Libraries

`.os`

`.pandas`

`.numpy`

`.matplotlib`

`.sklearn`

### Classes, Methods and Functions

**`SleepHealthLifestyleAnalyzer: `**This is the main class that contains all the functions for sleep health and lifestyle analysis. It has the following methods:

`__init__(file_path, features, target, test_size):` Initializes the SleepHealthLifestyleAnalyzer object with the file path, features, target variable, and test size.

`change_to_current_directory():` Changes the working directory to the current directory.

`load_data():` Loads the sleep health and lifestyle dataset from a CSV file.

`manipulate_data():` Manipulates the data by dropping rows with missing values and separating features and target variables.

`encode_target():` Encodes the target variable using label encoding.

`explore_data():` Prints the first few rows of the dataset and generates histograms of features.

`split_data():` Splits the data into training and testing sets.

`fit_logistic_regression():` Fits a logistic regression model to the training data and predicts on the testing data.

`fit_decision_tree():` Fits a decision tree classifier to the training data and predicts on the testing data.

`fit_random_forest():` Fits a random forest classifier to the training data and predicts on the testing data.

`fit_decision_tree_with_adaboost():` Fits an AdaBoost classifier with a decision tree base estimator to the training data and predicts on the testing data.

`fit_random_forest_with_adaboost():` Fits an AdaBoost classifier with a random forest base estimator to the training data and predicts on the testing data.

`evaluate_models():` Evaluates the performance of different models by comparing their predictions with the true labels.

`run():` Runs the sleep health and lifestyle analysis process, including loading the data, exploring it, fitting models, and evaluating their performance.

**`CustomAdaBoostClassifier:`** A custom implementation of the AdaBoost algorithm. It has the following methods:

`__init__(base_estimator=None, n_estimators=50):` Initializes the CustomAdaBoostClassifier object with a base estimator and the number of estimators.

`fit(X, y):` Fits the AdaBoost classifier to the training data.

`predict(X):` Predicts the target labels for the input features.

`predict_proba(X):` Predicts class probabilities for the input features.

### Execution

To execute the sleep health and lifestyle analysis using the terminal, follow these steps:

1-) Open a terminal or command prompt.

2-) Navigate to the directory where the script file is located using the cd command

3-) Ensure that the required libraries are installed. If not, install them using the following command:
Copy code : **`pip install -r requirements.txt`**

4-) Set the file path, features, and target variables in the __init__() method of the SleepHealthLifestyleAnalyzer class in the script file.

5-) Run the script using the following command:
Copy code : **`python sleep_health_lifestyle_analyzer.py`**

6-) Follow the instructions displayed in the terminal to explore the data and choose the model to run.

**Note:** Make sure to have the sleep health and lifestyle dataset (CSV file) in the same directory as the script.

Feel free to modify the code and parameters to suit your needs.






