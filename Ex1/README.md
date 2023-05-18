Authored By:
Hakam Nabulssi


Files Content:
LinearRegression.py: This file contains the Python code for implementing various optimization algorithms for linear regression.
README.md: This file contains instructions on how to use the code, a brief overview of the optimization algorithms, and a description of the file contents.
report.pdf: This file contains a report of the code, including a detailed explanation of the optimization algorithms, their advantages and disadvantages, and the results of experiments conducted using the code.


Brief Description:
This code contains implementations of three different gradient descent algorithms for linear regression: batch gradient descent, mini-batch gradient descent, and Adam optimizer.
These algorithms are commonly used in machine learning for optimizing the parameters of a model.

The code is written in Python and utilizes NumPy and Matplotlib libraries for numerical calculations and plotting.
The functions in this code can be used to fit a linear regression model on a given dataset and to compare the performance of different gradient descent algorithms.


Functions:
add_ones_column: adds a column of ones to the input matrix X. This is useful for linear regression as it allows for the inclusion of a bias term in the model.

predict: calculates the predicted value of y for a given input vector x and parameter vector theta.

cost_function: calculates the cost (mean squared error) of the linear regression model for a given set of parameters theta, input matrix X, and output vector y.

gradient: calculates the gradient of the cost function with respect to the parameter vector theta, input matrix X, and output vector y.

gradient_descent: implements batch gradient descent to optimize the parameters theta of the linear regression model.

mini_batch_gradient_descent: implements mini-batch gradient descent to optimize the parameters theta of the linear regression model.

adam_optimizer: implements the Adam optimizer to optimize the parameters theta of the linear regression model.

plot_optimizer_convergence: plots the convergence of the chosen optimizer for a range of learning rates.

main: reads a CSV file, performs data normalization, and runs various optimization algorithms to minimize the cost function


Executing Steps:
python <file_name>.py

**Note: When you run the code, you will be prompted to enter the path where the folder is located. Please make sure to enter the correct path in order for the code to work properly.

