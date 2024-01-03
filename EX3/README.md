Authors
Hakam Nabulssi 
Hanna Bajjaly 

K-means Clustering
This code implements the K-means clustering algorithm on a generated dataset with two groups of data points. It uses the scikit-learn library for K-means clustering and data preprocessing.

Code Structure
The code is organized as follows:

Importing the necessary libraries:

numpy for numerical computations.
matplotlib.pyplot for data visualization.
sklearn.cluster.KMeans for K-means clustering.
sklearn.metrics.silhouette_score for evaluating clustering performance.
sklearn.preprocessing.StandardScaler for data normalization.
Function generate_data():

Generates a dataset with two groups of data points.
Returns the normalized dataset.
Function data_normalization(data):

Normalizes the input dataset using StandardScaler.
Returns the normalized data.
Function plot_data(data):

Plots the given dataset.
Function fit_kmeans(data):

Fits the K-means algorithm on the given dataset.
Returns the trained KMeans object.
Function predict_labels(kmeans, data):

Predicts cluster labels for the given dataset using the trained KMeans model.
Returns the predicted cluster labels.
Function plot_results(data, predicted_labels):

Plots the results of the K-means clustering.
Function evaluate_algorithm(kmeans, data):

Evaluates the K-means algorithm by calculating the inertia score and silhouette score.
Function main():

Orchestrates the overall workflow:
Generates the dataset.
Plots the data.
Fits the K-means algorithm.
Predicts cluster labels.
Plots the results.
Evaluates the algorithm.
Conditional check if __name__ == '__main__'::

Calls the main() function to start the program execution.
Usage
To use this code, follow these steps:

Install the required libraries:

numpy
matplotlib
scikit-learn
Copy the code into a Python environment or save it as a Python script (e.g., kmeans_clustering.py).

Run the script. The generated dataset will be plotted, followed by the K-means clustering results and evaluation scores.

Dependencies
This code requires the following dependencies:

numpy
matplotlib
scikit-learn
Ensure that these libraries are installed in your Python environment before running the code.

Contributing
Contributions to this code are welcome. Feel free to suggest improvements or report any issues.

