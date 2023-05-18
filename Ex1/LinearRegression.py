import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def add_ones_column(X):
    """
        Adds a column of ones to the beginning of a 2D numpy array.

        Parameters:
            X : The 2D numpy array to add the ones column to.

        Returns:
            numpy array: The modified numpy array with a column of ones added to the beginning.
        """
    ones_column = np.ones((X.shape[0], 1))
    X_with_ones = np.hstack((ones_column, X))
    return X_with_ones


def predict(x, theta):
    """
       Predict the output for a given input and model parameters.

       Parameters:
       x : An array containing the input values.
       theta : An array containing the model parameters.

       Returns:
       An array containing the predicted output values.
       """
    return np.dot(x, theta)


def cost_function(theta, X, y):
    """
        Calculates the cost function J for a given hypothesis function theta, input X, and expected output y.

        Parameters:
        theta : An array of parameters for the hypothesis function.
        X : A matrix of input features, where each row represents a single training example and each column represents a single feature.
        y : A vector of expected outputs for each training example.

        Returns:
        float: The calculated cost function J for the given inputs.
        """
    m = len(y)
    h = predict(X, theta)
    J = (1 / (2 * m)) * np.sum(np.square(h - y))
    return J


def gradient(theta, X, y):
    """
       Computes the gradient of the cost function for linear regression.

       Parameters:
       theta: Parameters to be learned.
       X : Matrix of features.
       y : Vector of target values.

       Returns:
       grad (numpy array): Vector of partial derivatives of the cost function.
       """
    m = len(y)
    h = predict(X, theta)
    grad = (1 / m) * np.dot(X.T, (h - y))
    return grad


def gradient_descent(X, y, theta, alpha, num_iters):
    """
      Gradient descent algorithm for linear regression.

      Parameters:
      X : The feature matrix of shape (m, n+1), where m is the number of examples and n is the number of features.
          The first column of X should be all ones (bias term).
      y : The target vector of shape (m,).
      theta : The parameter vector of shape (n+1,).
      alpha : The learning rate.
      num_iters : The number of iterations.

      Returns:
      theta : The optimal parameter vector of shape (n+1,).
      J_history : The cost function history of shape (num_iters,).
      """
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - alpha * gradient(theta, X, y)
        J_history[i] = cost_function(theta, X, y)
    return theta, J_history


def mini_batch_gradient_descent(X, y, theta, alpha, num_iters, batch_size):
    """
        Performs mini-batch gradient descent on a given dataset with a specified batch size.

        Parameters:
        X : Features matrix of shape (m, n+1), where m is the number of examples and n is the number of features.
        y : Target variable vector of shape (m, 1).
        theta : Parameter vector of shape (n+1, 1).
        alpha : Learning rate.
        num_iters : Number of iterations to run the algorithm.
        batch_size : Number of examples to include in each batch.

        Returns:
        A tuple containing the optimized parameter vector and the history of the cost function over iterations.

        """
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        idx = np.random.choice(m, batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        theta = theta - alpha * gradient(theta, X_batch, y_batch)
        J_history[i] = cost_function(theta, X, y)
    return theta, J_history


def adam_optimizer(X, y, theta, alpha, num_iters, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
        This function performs parameter optimization using the ADAM algorithm.

        Parameters:
        X : The input features of shape (m, n)
        y : The target values of shape (m,)
        theta : The initial parameter vector of shape (n,)
        alpha : The learning rate
        num_iters : The number of iterations
        beta1 : The exponential decay rate for the first moment estimates. Default is 0.9
        beta2 : The exponential decay rate for the second moment estimates. Default is 0.999
        epsilon : A small value to prevent division by zero. Default is 1e-8.

        Returns:
        theta : The optimized parameter vector of shape (n,)
        J_history : The cost history of shape (num_iters,)

        """
    J_history = np.zeros(num_iters)
    v = np.zeros(X.shape[1])
    s = np.zeros(X.shape[1])

    for i in range(num_iters):
        grad = gradient(theta, X, y)
        v = beta1 * v + (1 - beta1) * grad
        s = beta2 * s + (1 - beta2) * np.square(grad)
        v_hat = v / (1 - beta1 ** (i + 1))
        s_hat = s / (1 - beta2 ** (i + 1))
        theta = theta - alpha * v_hat / (np.sqrt(s_hat) + epsilon)
        J_history[i] = cost_function(theta, X, y)

    return theta, J_history


def plot_optimizer_convergence(X, y, theta_init, optimizer, learning_rates, num_iters, batch_size=None, beta1=0.9,
                               beta2=0.999, epsilon=1e-8):
    """
        Plots the convergence of different optimization algorithms.

        Parameters:
        X : Input features.
        y : Output labels.
        theta_init : Initial values of model parameters.
        optimizer : Type of optimization algorithm to use. Must be one of 'gd', 'mbgd', or 'adam'.
        learning_rates : List of learning rates to use for optimization.
        num_iters : Number of iterations for optimization.
        batch_size : Number of samples per batch for mini-batch gradient descent. Default is None.
        beta1 : Exponential decay rate for the first moment estimates in Adam optimizer. Default is 0.9.
        beta2 : Exponential decay rate for the second moment estimates in Adam optimizer. Default is 0.999.
        epsilon: A small value to avoid division by zero in Adam optimizer. Default is 1e-8.

        """
    title = ''
    for alpha in learning_rates:
        if optimizer == 'gd':
            theta, J_history = gradient_descent(X, y, theta_init, alpha, num_iters)
            # print(f'Gradient descent with alpha={alpha}: theta={theta}, cost={J_history[-1]}')
            plt.plot(np.arange(num_iters), J_history, label=f'alpha={alpha}')
            title = 'Convergence of gradient descent'
            print(f"**Gradient Descent Succeeded for {alpha}  \n")
        elif optimizer == 'mbgd':
            if batch_size is None:
                raise ValueError("batch_size cannot be None for mini-batch gradient descent")
            theta, J_history = mini_batch_gradient_descent(X, y, theta_init, alpha, num_iters, batch_size)
            # print(f'MINI BATCH with alpha={alpha}: theta={theta}, cost={J_history[-1]}')
            plt.plot(np.arange(num_iters), J_history, label=f'alpha={alpha}, batch_size={batch_size}')
            title = 'Convergence of mini-batch gradient descent'
            print(f"**Mini Batch Gradient Descent Succeeded for {alpha} \n")
        elif optimizer == 'adam':
            theta, J_history = adam_optimizer(X, y, theta_init, alpha, num_iters, beta1, beta2, epsilon)
            # print(f'ADAM OPTIMIZER with alpha={alpha}: theta={theta}, cost={J_history[-1]}')
            plt.plot(np.arange(num_iters), J_history, label=f'alpha={alpha}')
            title = 'Convergence of Adam optimizer'
            print(f" **Adam Optimizer Algorithm Succeeded for {alpha} \n")
        else:
            raise ValueError("Invalid optimizer type. Must be one of 'gd', 'mbgd', or 'adam'")

    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.title(title)
    plt.legend()
    plt.show()


# This function reads a CSV file, performs data normalization, and runs various optimization algorithms to minimize
# the cost function
def main():
    # Prompt the user to enter the path of the CSV file
    path = input(
        "Please enter the path to the file in the following format: <folder_path> <name_of_the_database_file> (e.g. /home/user/Folder/cancer_data.csv): ")

    # Load the database into a Pandas DataFrame
    data = pd.read_csv(path, header=None)
    column_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y']
    data.columns = column_names

    print("\n************************** This Is The DataFrame That You Are Using ! **************************\n")
    print(data)

    # Create matrices X and y
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    print("\n\nMean before data normalization: ", mean)
    print("Standard deviation before data normalization:", std)
    print("\n")

    # Normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    std = np.std(X, axis=0)
    mean = np.mean(X, axis=0)

    mean[np.abs(mean) < 1e-13] = 0

    if np.allclose(mean, 0) and np.allclose(std, 1):
        print("Data normalized successfully!")
        print("Mean of the normalized data: ", mean)
        print("Standard deviation of the normalized data: ", np.std(X, axis=0))
    else:
        print("Data normalization failed.\n the program will exit !")
        exit(1)

    # Add a column of ones to matrix X
    X = add_ones_column(X)
    # Check if the first column of X is equal to ones
    if np.all(X[:, 0] == 1):
        print("\nIntercept term added successfully, The first column of X is equal to ones.\n")

    else:
        print("\nIntercept term failed, The first column of X is not equal to ones.\n the program will exit ! \n")
        exit(1)

    print("************************** Running Gradient Descent **************************\n")
    plot_optimizer_convergence(X, y, np.random.rand(X.shape[1]), 'gd', [0.5, 0.1, 0.01, 0.001], 1000)

    print("************************** Mini Batch Gradient Descent **************************\n")
    plot_optimizer_convergence(X, y, np.random.rand(X.shape[1]), 'mbgd', [0.5, 0.1, 0.01, 0.001], 1000,
                               batch_size=random.randint(32, 256))

    print("************************** Adam Optimizer Algorithm **************************\n")
    plot_optimizer_convergence(X, y, np.random.rand(X.shape[1]), 'adam', [0.5, 0.1, 0.01, 0.001], 1000, beta1=0.95,
                               beta2=0.999, epsilon=1e-6)

    print("Program successfully Executed")


# calling the main function to execute the code
if __name__ == "__main__":
    main()
