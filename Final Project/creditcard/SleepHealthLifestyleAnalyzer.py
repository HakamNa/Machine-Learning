import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
)


class CustomAdaBoostClassifier:
    """
    A custom implementation of AdaBoost classifier.

    """

    def __init__(self, base_estimator=None, n_estimators=50):
        self.estimator_weights_ = None
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    @staticmethod
    def _fit_estimator(estimator, X, y, sample_weights):
        """
        Fits the given estimator on the training data with weighted samples.

        """
        if hasattr(estimator, 'fit'):
            estimator.fit(X, y, sample_weight=sample_weights)
        else:
            raise AttributeError("Base estimator does not have 'fit' method.")

    @staticmethod
    def _predict_estimator(estimator, X):
        """
        Predicts using the given estimator.

        """
        if hasattr(estimator, 'predict'):
            return estimator.predict(X)
        else:
            raise AttributeError("Base estimator does not have 'predict' method.")

    def fit(self, X, y):
        """
        Fits the AdaBoost classifier on the training data.

        """
        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples

        self.estimator_weights_ = []  # Initialize the estimator_weights_ attribute as an empty list

        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            self._fit_estimator(estimator, X, y, sample_weights)
            y_pred = self._predict_estimator(estimator, X)

            err = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)

            if err >= 0.5:  # Modified condition to handle cases where err == 0.5
                break

            alpha = 0.5 * np.log((1.0 - err) / err)

            sample_weights *= np.exp(-alpha * y * y_pred)
            sample_weights /= np.sum(sample_weights)

            self.estimators.append(estimator)
            self.estimator_weights_.append(alpha)  # Append the alpha to estimator_weights_

        # Ensure the number of estimator weights matches the number of fitted estimators
        num_fitted_estimators = len(self.estimators)
        num_estimator_weights = len(self.estimator_weights_)
        if num_fitted_estimators < num_estimator_weights:
            self.estimator_weights_ = self.estimator_weights_[:num_fitted_estimators]
        elif num_fitted_estimators > num_estimator_weights:
            self.estimator_weights_ += [0.0] * (num_fitted_estimators - num_estimator_weights)

        # If no fitted estimators, set estimator_weights_ to an empty list
        if not self.estimators:
            self.estimator_weights_ = []

    def predict(self, X):
        """
        Predicts the target labels using the fitted AdaBoost classifier.

        """
        y_pred = np.zeros(len(X))
        for i, estimator in enumerate(self.estimators):
            if i < len(self.estimator_weights):
                y_pred += self.estimator_weights[i] * self._predict_estimator(estimator, X)
            else:
                break

        return np.sign(y_pred)

    def predict_proba(self, X):
        """
        Predicts class probabilities for binary classification.

        """
        n_samples = len(X)
        proba = np.zeros((n_samples, 2))
        for i, estimator in enumerate(self.estimators):
            if i < len(self.estimator_weights):
                if hasattr(estimator, 'predict_proba'):
                    proba_estimator = estimator.predict_proba(X)
                else:
                    raise AttributeError("Base estimator does not have 'predict_proba' method.")

                proba += self.estimator_weights[i] * proba_estimator

        # Check if the sum of estimator weights is not zero before dividing
        if np.sum(self.estimator_weights) != 0:
            proba /= np.sum(self.estimator_weights)

        return proba


class SleepHealthLifestyleAnalyzer:
    def __init__(self, file_path, features, target, test_size=0.2):
        self.file_path = file_path
        self.features = features
        self.target = target
        self.test_size = test_size
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_labels_encoder = None

    @staticmethod
    def change_to_current_directory():
        """
        Changes the working directory to the current directory.
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_directory)

    def load_data(self):
        """
        Load the Sleep_health_and_lifestyle_dataset from a CSV file.
        """
        self.data = pd.read_csv(self.file_path)

    def manipulate_data(self):
        # Drop rows with missing values
        self.data = self.data.dropna()

        # Exclude the 'Person ID' column from features
        features_without_id = [col for col in self.features if col != 'Person ID']
        self.X = self.data[features_without_id]
        self.y = self.data[self.target]

        # One-hot encode the 'Gender', 'Occupation', 'BMI Category', and 'Sleep Disorder' columns
        self.X = pd.get_dummies(self.X, columns=['Gender', 'Occupation', 'Blood Pressure', 'BMI Category'],
                                drop_first=True)

        # Get the columns before scaling to use them later
        columns = self.X.columns

        # Normalize the features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        # Convert the scaled features back to DataFrame
        self.X = pd.DataFrame(self.X, columns=columns)


    def encode_target(self):
        """
        Encode the target variable using LabelEncoder and fit the encoder with target labels.
        """
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)  # Fit and transform the target labels
        self.target_labels_encoder = le  # Store the encoder to use it for decoding later

    def explore_data(self):
        """
        Explore the data by printing the first few rows and generating histograms.
        """
        features_to_plot = self.data.columns[1:]  # Exclude the first column (ID)

        print(self.data.head())  # Print the first few rows of the dataset
        print(self.data.describe())  # Get statistical summary of the dataset

        # Plotting histograms of features
        self.data[features_to_plot].hist(figsize=(10, 10))
        plt.tight_layout()
        plt.show()

    def split_data(self):
        """
        Split the data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=None)

    def fit_logistic_regression(self):
        """
        Fit a logistic regression model and return the predicted values and probabilities.
        """
        logistic_reg = LogisticRegression()
        logistic_reg.fit(self.X_train, self.y_train)
        y_pred = logistic_reg.predict(self.X_test)
        y_scores = logistic_reg.predict_proba(self.X_test)[:, 1]

        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Logistic Regression: Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_scores)
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Logistic Regression: Precision-Recall Curve')
        plt.legend(loc='lower right')
        plt.show()

        return y_pred, y_scores

    def fit_decision_tree(self, feature_names=None, max_depth=None):
        """
        Fit a Decision Tree classifier on the whole dataset and plot the decision tree and feature importances.
        """
        decision_tree = DecisionTreeClassifier(max_depth=max_depth)
        decision_tree.fit(self.X, self.y)

        # Plot the Decision Tree
        plt.figure(figsize=(12, 8))
        plot_tree(decision_tree, feature_names=feature_names, class_names=list(self.target_labels_encoder.classes_),
                  filled=True, rounded=True)
        plt.title("Decision Tree")
        plt.show()

        # Feature importance plot
        importances = decision_tree.feature_importances_
        plt.barh(self.X.columns, importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Decision Tree: Feature Importance')
        plt.show()

        return decision_tree.predict(self.X), decision_tree.predict_proba(self.X)[:, 1]

    def fit_random_forest(self):
        """
        Fit a random forest classifier and return the predicted values and probabilities.
        """
        random_forest = RandomForestClassifier()
        random_forest.fit(self.X_train, self.y_train)
        y_pred = random_forest.predict(self.X_test)
        y_scores = random_forest.predict_proba(self.X_test)[:, 1]

        # Feature importance plot
        importances = random_forest.feature_importances_
        plt.barh(self.X_train.columns, importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Random Forest: Feature Importance')
        plt.show()

        # ROC curve using predicted probabilities
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Random Forest: Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        return y_pred, y_scores

    def fit_decision_tree_with_adaboost(self):
        # Fit a decision tree classifier
        decision_tree = DecisionTreeClassifier(max_depth=None)
        decision_tree.fit(self.X_train, self.y_train)

        # Fit the AdaBoost classifier using the decision tree as the base estimator
        adaboost_dt = CustomAdaBoostClassifier(base_estimator=decision_tree, n_estimators=500)
        adaboost_dt.fit(self.X_train, self.y_train)
        y_pred = adaboost_dt.predict(self.X_test)
        y_scores = adaboost_dt.predict_proba(self.X_test)[:, 1]

        # Iteration-wise error plot
        errors = []
        for y_pred_stage in adaboost_dt.estimators:
            error = np.mean(y_pred_stage.predict(self.X_test) != self.y_test)
            errors.append(error)
        plt.plot(range(1, len(errors) + 1), errors, marker='o')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Error Rate')
        plt.title('AdaBoost (Decision Tree): Iteration-wise Error Rate')
        plt.show()

        return y_pred, y_scores

    def fit_random_forest_with_adaboost(self):
        adaboost_rf = AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=100)
        adaboost_rf.fit(self.X_train, self.y_train)
        y_pred = adaboost_rf.predict(self.X_test)
        y_scores = adaboost_rf.predict_proba(self.X_test)[:, 1]

        # Iteration-wise error plot
        errors = []
        for y_pred_stage in adaboost_rf.staged_predict(self.X_test):
            errors.append(1 - accuracy_score(self.y_test, y_pred_stage))
        plt.plot(range(1, len(errors) + 1), errors, marker='o')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Error Rate')
        plt.title('AdaBoost (Random Forest): Iteration-wise Error Rate')
        plt.show()

        # Feature importance plot
        importances = np.mean([tree.feature_importances_ for tree in adaboost_rf.estimators_], axis=0)
        plt.barh(self.X_train.columns, importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('AdaBoost (Random Forest): Feature Importance')
        plt.show()

        return y_pred, y_scores

    @staticmethod
    def evaluate_models(y_true, *preds):
        """
        Evaluate the models by calculating and printing relevant metrics.
        """
        for idx, pred in enumerate(preds):
            if isinstance(pred[0], (int, np.integer)):  # Classification models
                accuracy = accuracy_score(y_true, pred)
                print(f"Model {idx + 1} Accuracy: {accuracy:.4f}")

                # Additional metrics for classification models
                if np.unique(pred).size == 2:  # Check if it's a binary classification
                    pred = np.where(pred == -1, 0, 1)  # Convert -1 to 0 for binary evaluation
                cm = confusion_matrix(y_true, pred)
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * (precision * recall) / (precision + recall)
                roc_auc = roc_auc_score(y_true, pred)
                print(f"Model {idx + 1} Confusion Matrix:")
                print(cm)
                print(f"Model {idx + 1} Precision: {precision:.4f}")
                print(f"Model {idx + 1} Recall: {recall:.4f}")
                print(f"Model {idx + 1} F1-Score: {f1_score:.4f}")
                print(f"Model {idx + 1} ROC AUC: {roc_auc:.4f}")
            else:
                pass

    def run(self):
        """
        Run the analysis pipeline.
        """
        self.change_to_current_directory()
        self.load_data()

        explore_choice = input(
            "\nWould you like to view summary information about the data after it has been manipulated? (y/n)"
        )
        if explore_choice.lower() == 'y':
            self.explore_data()

        self.manipulate_data()
        self.encode_target()
        self.split_data()

        while True:
            print("\nAVAILABLE MODELS:")
            print("1. Logistic Regression")
            print("2. Decision Tree")
            print("3. Random Forest")
            print("4. Manually Implemented AdaBoost (Decision Tree)")
            print("5. AdaBoost using Python adaboost library (Random Forest)")
            print("-1. Exit")

            model_choice = input("\nChoose the algorithm you want to run (1-5), or enter -1 to exit: ")

            if model_choice == "-1":
                print("Exiting the program...")
                break
            elif model_choice == "1":
                print("Running Logistic Regression, please wait a moment ...")
                logistic_reg_pred, logistic_reg_scores = self.fit_logistic_regression()
                self.evaluate_models(self.y_test, logistic_reg_pred)
            elif model_choice == "2":
                print("Running Decision Tree, please wait a moment ...")
                # Call the fit_decision_tree method and pass feature_names and max_depth arguments
                decision_tree_pred, decision_tree_scores = self.fit_decision_tree(
                    feature_names=self.X.columns.tolist(),
                    max_depth=None  # You can set max_depth to a specific value if you want to limit the tree depth
                )
                # Evaluate the decision tree
                self.evaluate_models(self.y, decision_tree_pred)
            elif model_choice == "3":
                print("Running Random Forest, please wait a moment ...")
                random_forest_pred, random_forest_scores = self.fit_random_forest()
                self.evaluate_models(self.y_test, random_forest_pred)
            elif model_choice == "4":
                print("Running Manual AdaBoost with Decision Tree, please wait a moment ...")
                adaboost_dt_pred, adaboost_dt_scores = self.fit_decision_tree_with_adaboost()
                self.evaluate_models(self.y_test, adaboost_dt_pred)
            elif model_choice == "5":
                print("Running AdaBoost using python library with Random Forest, please wait a moment ...")
                adaboost_rf_pred, adaboost_rf_scores = self.fit_random_forest_with_adaboost()
                self.evaluate_models(self.y_test, adaboost_rf_pred)
            else:
                print("Invalid choice. Please enter a valid choice (1-5 or -1).")


def main():
    file_name = 'Sleep_health_and_lifestyle_dataset.csv'
    file_path = os.path.join(os.getcwd(), file_name)

    # Exclude 'Person ID' column from features
    features = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                'Stress Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps']

    target = 'Sleep Disorder'

    analyzer = SleepHealthLifestyleAnalyzer(file_path, features, target)
    analyzer.run()


if __name__ == '__main__':
    main()
