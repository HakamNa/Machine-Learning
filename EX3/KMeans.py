

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def generate_data():
    """
    Generate the dataset with two groups of data points.

    Returns:
        normalized_data: The generated and normalized dataset.
    """
    np.random.seed(None)  # Set random seed

    group1_mean = [-1, -1]
    group1_cov = [[0.8, 0], [0, 0.8]]
    group1_size = 700
    group1_data = np.random.multivariate_normal(group1_mean, group1_cov, group1_size)

    group2_mean = [1, 1]
    group2_cov = [[0.75, -0.2], [-0.2, 0.6]]
    group2_size = 300
    group2_data = np.random.multivariate_normal(group2_mean, group2_cov, group2_size)

    data = np.concatenate((group1_data, group2_data))

    normalized_data = data_normalization(data)

    return normalized_data


def data_normalization(data):
    """
    Normalizing the input dataset.

    Parameters:
        data: The input dataset.

    Returns:
        normalized_data : The normalized data .
    """
    # Normalize the dataset using StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    return normalized_data


def plot_data(data):
    """
    Plot the given dataset.

    Parameters:
        data: The dataset to be plotted.
    """
    colors = ['black']
    plt.scatter(data[:, 0], data[:, 1], color='blue')
    plt.title("Generated Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def fit_kmeans(data):
    """
    Fit the k-means algorithm on the given dataset.

    Parameters:
        data: The input dataset.

    Returns:
        kmeans: The trained KMeans object.
    """
    kmeans = KMeans(n_clusters=2, init='random', max_iter=1000)
    kmeans.fit(data)
    return kmeans


def predict_labels(kmeans, data):
    """
    Predict cluster labels for the given dataset using the trained KMeans model.

    Parameters:
        kmeans: The trained KMeans object.
        data: The input dataset.

    Returns:
        predicted_labels: The predicted cluster labels.
    """
    predicted_labels = kmeans.predict(data)
    return predicted_labels


def plot_results(data, predicted_labels):
    """
    Plot the results of the K-means clustering.

    Parameters:
        data: The input dataset.
        predicted_labels: The predicted cluster labels.
    """
    colors = ['red', 'green']
    plt.scatter(data[:, 0], data[:, 1], c=[colors[label] for label in predicted_labels])
    plt.title("K-means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def evaluate_algorithm(kmeans, data):
    """
    Evaluate the K-means algorithm by calculating the inertia score and silhouette score.

    Parameters:
        kmeans: The trained KMeans object.
        data: The input dataset.
    """
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(data, labels)
    if silhouette_avg >= 0.85:
        print("The k-means score is 85 or higher.")
    else:
        print("The k-means score is less than 85.")

    if kmeans.n_iter_ < kmeans.max_iter:
        print("The k-means algorithm has converged.")
    else:
        print("The k-means algorithm has not converged.")


def main():
    """
    . Generate the dataset
    . Plot the data
    . Fit the k-means algorithm
    . Predict cluster labels
    . Plot the results
    . Evaluate the algorithm
    """
    data = generate_data()

    plot_data(data)

    kmeans = fit_kmeans(data)

    predicted_labels = predict_labels(kmeans, data)

    plot_results(data, predicted_labels)

    evaluate_algorithm(kmeans, data)


if __name__ == '__main__':
    """
    Calling the main funciton
    """
    main()
