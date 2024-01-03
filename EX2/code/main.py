import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


def change_to_current_directory():
    """
    Changes the working directory to the current directory.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_directory)


def read_input_data():
    """
    Reads the input data from the CSV files.

    Returns:
        X : Input features.
        Y : Target labels.
    """
    X = pd.read_csv('ex2_x_data.csv')
    Y = pd.read_csv('ex2_y_data.csv')
    return X, Y


def shuffle_data(X, Y):
    """
    Shuffles the input data.

    Parameters:
        X : Input features.
        Y : Target labels.

    Returns:
        X_shuffled : Shuffled input features.
        Y_shuffled : Shuffled target labels.
    """

    # Create a random permutation of indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Shuffle X and Y using the generated indices
    X_shuffled = X.iloc[indices].reset_index(drop=True)
    Y_shuffled = Y.iloc[indices].reset_index(drop=True)

    return X_shuffled, Y_shuffled


def split_data(X, Y):
    """
    Splits the data into training and testing sets.

    Parameters:
        X : Input features.
        Y : Target labels.

    Returns:
        X_train : Training set input features.
        X_test : Testing set input features.
        y_train : Training set target labels.
        y_test : Testing set target labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Creates and trains the logistic regression model.

    Parameters:
        X_train : Training set input features.
        y_train : Training set target labels.

    Returns:
        model : Trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def predict_labels(model, X):
    """
    Predicts the labels for the data.

    Parameters:
        model : Trained logistic regression model.
        X : Input features.

    Returns:
        y_pred : Predicted labels.
    """
    return model.predict(X)


def evaluate_model(y_true, y_pred):
    """
    Evaluates the model using the confusion matrix.

    Parameters:
        y_true : True labels.
        y_pred : Predicted labels.

    Returns:
        confusion_matrix : Confusion matrix.
    """
    confusion_matrix = pd.crosstab(y_true.values.ravel(), y_pred, rownames=['Actual'], colnames=['Predicted'])
    return confusion_matrix


def plot_confusion_matrices(train_confusion_matrix, test_confusion_matrix):
    """
    Plots the confusion matrices for training and testing data.

    Parameters:
        train_confusion_matrix : Confusion matrix for training data.
        test_confusion_matrix : Confusion matrix for testing data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Training data confusion matrix
    image1 = axes[0].imshow(train_confusion_matrix, cmap='Blues')
    axes[0].set_title('Confusion Matrix - Training Data')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    plt.colorbar(image1, ax=axes[0])

    # Test data confusion matrix
    image2 = axes[1].imshow(test_confusion_matrix, cmap='Oranges')
    axes[1].set_title('Confusion Matrix - Test Data')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    plt.colorbar(image2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def calculate_metrics(confusion_matrix):
    """
    Calculates precision, recall, and F1 score based on the confusion matrix.

    Parameters:
        confusion_matrix : Confusion matrix.

    Returns:
        precision : Precision score.
        recall : Recall score.
        f1_score : F1 score.
    """
    true_positives = confusion_matrix.loc[1.0, 1.0]
    false_positives = confusion_matrix.loc[0.0, 1.0]
    false_negatives = confusion_matrix.loc[1.0, 0.0]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def performance_analysis(train_matrix, test_matrix):
    """
    Performs additional analysis on the model's performance.

    Parameters:
        train_matrix : Confusion matrix for the training data.
        test_matrix : Confusion matrix for the test data.

    Returns:
        accuracy_train : Accuracy score for the training data.
        accuracy_test : Accuracy score for the test data.
        precision_train : Precision score for the training data.
        recall_train : Recall score for the training data.
        f1_score_train : F1 score for the training data.
        precision_test : Precision score for the test data.
        recall_test : Recall score for the test data.
        f1_score_test : F1 score for the test data.
    """
    accuracy_train = (train_matrix.loc[0.0, 0.0] + train_matrix.loc[1.0, 1.0]) / train_matrix.values.sum()
    accuracy_test = (test_matrix.loc[0.0, 0.0] + test_matrix.loc[1.0, 1.0]) / test_matrix.values.sum()

    precision_train, recall_train, f1_score_train = calculate_metrics(train_matrix)
    precision_test, recall_test, f1_score_test = calculate_metrics(test_matrix)

    return (
        accuracy_train,
        accuracy_test,
        precision_train,
        recall_train,
        f1_score_train,
        precision_test,
        recall_test,
        f1_score_test
    )


def main():
    """
    Main function to execute the logistic regression model.

    This function performs the following steps:
    . Changes the current directory.
    . Reads the input data.
    . Shuffles the input data.
    . Splits the data into training and testing sets.
    . Trains the logistic regression model.
    . Predicts the labels for the training and test data.
    . Evaluates the model using confusion matrices.
    . Calculating the Accuracy for the training and the testing set.
    . Prints and plots the confusion matrices.

    """
    change_to_current_directory()

    # Read the input data
    X, Y = read_input_data()
    print("********* Input Data *********")
    print("X:\n", X)
    print("Y:\n", Y)

    # Shuffle the input data
    X, Y = shuffle_data(X, Y)
    print("\n********Data Shuffled********")
    print("Shuffled X:\n", X)
    print("Shuffle Y:\n", Y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, Y)
    print("********Data Split********")
    print("Training data - X_train:\n", X_train)
    print("Training labels - y_train:\n", y_train)
    print("Testing data - X_test:\n", X_test)
    print("Testing labels - y_test:\n", y_test)

    # Train the logistic regression model
    model = train_model(X_train, y_train)
    print("*******Model Training*******")
    print("Logistic Regression model trained.")

    # Predict the labels for the training data
    y_train_pred = predict_labels(model, X_train)
    print("*******Predictions on Training Data*******")
    print("Predicted labels for the training data:\n", y_train_pred)

    # Predict the labels for the test data
    y_test_predict = predict_labels(model, X_test)
    print("*******Predictions on Test Data*******")
    print("Predicted labels for the test data:\n", y_test_predict)

    # Evaluate the model (Confusion matrices)
    train_confusion_matrix = evaluate_model(y_train, y_train_pred)
    test_confusion_matrix = evaluate_model(y_test, y_test_predict)

    # Print confusion matrices
    print("Confusion Matrix - Training Data:\n", train_confusion_matrix)
    print("Confusion Matrix - Test Data:\n", test_confusion_matrix)

    # Perform additional analysis
    print(performance_analysis(train_confusion_matrix, test_confusion_matrix))

    # Plot confusion matrices
    plot_confusion_matrices(train_confusion_matrix, test_confusion_matrix)


if __name__ == "__main__":
    """
    Entry point of the program.
    """
    main()




