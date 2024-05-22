import random

import matplotlib as mpl
import matplotlib.pyplot as plt

# numpy
import numpy as np

# pytorch
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader

def tensorGenReg(num_examples=1000, w=[2, -1, 1], bias=True, delta=0.01, deg=1):
    """
    Generate a regression dataset with options for polynomial degrees and bias.

    This function generates random features and calculates their associated labels
    using a polynomial model of a specified degree. Noise is added to the labels
    based on a Gaussian distribution.

    Parameters:
    - num_examples (int): Number of examples in the dataset.
    - w (list): Coefficients of the regression model. Includes the bias term if 'bias' is True.
    - bias (bool): If True, the last element in 'w' is considered as the bias (intercept).
    - delta (float): Standard deviation of the Gaussian noise added to the labels.
    - deg (int): Degree of the polynomial model to be applied to the features.

    Returns:
    - tuple: A tuple containing two tensors:
      - features (torch.Tensor): The feature matrix (num_examples, num_features).
      - labels (torch.Tensor): The labels tensor (num_examples, 1) with noise.
    """
    num_inputs = len(w) - 1 if bias else len(w)
    features = torch.randn(num_examples, num_inputs)

    if bias:
        w_true = torch.tensor(w[:-1], dtype=torch.float32).reshape(-1, 1)
        b_true = w[-1]
    else:
        w_true = torch.tensor(w, dtype=torch.float32).reshape(-1, 1)
        b_true = 0

    # Calculate polynomial features if needed
    polynomial_features = torch.pow(features, deg)
    
    # Compute the labels using matrix multiplication for polynomial features and adding bias
    labels_true = polynomial_features @ w_true + b_true
    
    # Add Gaussian noise to labels
    labels = labels_true + torch.randn(labels_true.shape) * delta
    
    if bias:
        # Append a column of ones to features for bias
        features = torch.cat([features, torch.ones(num_examples, 1)], dim=1)
    
    return features, labels

def tensorGenCla(num_examples=500, num_inputs=2, num_class=3, deg_dispersion=[4, 2], bias=False):
    """
    Generate a classification dataset with normally distributed features.

    This function creates a dataset for classification problems, generating features and labels
    for a specified number of classes. Features are generated with a normal distribution where
    the mean of each class is linearly spaced and scaled by the deg_dispersion parameter.

    Parameters:
    - num_examples (int): Number of examples per class.
    - num_inputs (int): Number of features per example.
    - num_class (int): Number of classes.
    - deg_dispersion (list of two ints): Dispersion parameters for the dataset:
      - First element is the scale factor for the means of each class.
      - Second element is the standard deviation for the feature distributions.
    - bias (bool): If True, an additional bias term (column of ones) is added to the feature tensor.

    Returns:
    - tuple of torch.Tensor: (features, labels)
      - features (torch.Tensor): The feature tensor of shape (num_examples*num_class, num_inputs [+1 if bias]).
      - labels (torch.Tensor): The labels tensor of shape (num_examples*num_class, 1), containing class indices.
    """
    
    mean_scale, std_dev = deg_dispersion
    k = mean_scale * (num_class - 1) / 2
    
    features_list = []
    labels_list = []

    for class_index in range(num_class):
        # Calculate mean for current class
        mean_current = class_index * mean_scale - k
        # Generate features for current class
        class_features = torch.normal(mean=mean_current, std=std_dev, size=(num_examples, num_inputs))
        # Generate labels for current class
        class_labels = torch.full((num_examples, 1), class_index, dtype=torch.float32)
        
        features_list.append(class_features)
        labels_list.append(class_labels)

    # Concatenate all class features and labels
    features = torch.cat(features_list, dim=0).float()
    labels = torch.cat(labels_list, dim=0)

    if bias:
        # Append a column of ones to the features for the bias term
        bias_column = torch.ones(num_examples * num_class, 1)
        features = torch.cat((features, bias_column), dim=1)

    return features, labels



def data_iter(batch_size, features, labels):
    """
    Iteratively yield batches of features and labels from the provided datasets.

    This function shuffles the indices of the provided datasets and yields batches of the specified size,
    ensuring that each batch contains a subset of features and corresponding labels. If the total number of examples
    is not divisible by the batch size, the last batch will contain the remaining elements.

    Parameters:
    - batch_size (int): The number of examples per batch.
    - features (torch.Tensor): The input features tensor, expected to be 2D (num_examples, num_features).
    - labels (torch.Tensor): The input labels tensor, aligned with features by the first dimension.

    Returns:
    - generator: A generator that yields tuples (feature_batch, label_batch) where:
      - feature_batch (torch.Tensor): A tensor slice containing a batch of features.
      - label_batch (torch.Tensor): A tensor slice containing a batch of labels corresponding to the feature_batch.
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # Shuffle indices to randomize the data batches
    
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield (torch.index_select(features, 0, batch_indices), 
               torch.index_select(labels, 0, batch_indices))

def data_iter_list(batch_size, features, labels):
    """
    Generates a list of batches of features and labels from the provided datasets.

    This function shuffles the indices of the provided datasets and creates a list where each element is a tuple
    containing a batch of the specified size of features and corresponding labels. If the total number of examples
    is not divisible by the batch size, the last batch will contain the remaining elements.

    Parameters:
    - batch_size (int): The number of examples per batch.
    - features (torch.Tensor): The input features tensor, expected to be 2D (num_examples, num_features).
    - labels (torch.Tensor): The input labels tensor, aligned with features by the first dimension.

    Returns:
    - list of tuples: Each tuple contains two torch.Tensors:
      - feature_batch (torch.Tensor): A tensor slice containing a batch of features.
      - label_batch (torch.Tensor): A tensor slice containing a batch of labels corresponding to the feature_batch.
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # Shuffle indices to randomize the data batches
    batches = []  # List to store the batches
    
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        feature_batch = torch.index_select(features, 0, batch_indices)
        label_batch = torch.index_select(labels, 0, batch_indices)
        batches.append((feature_batch, label_batch))
    
    return batches

def linreg(X, w):
    """
    Perform linear regression computation.

    This function calculates the predicted values (y_hat) using the linear regression model
    y_hat = Xw, where X is the matrix of input features, and w is the vector of weights.

    Parameters:
    - X (torch.Tensor): The input feature matrix of size (n_samples, n_features).
    - w (torch.Tensor): The weight vector of size (n_features, 1).

    Returns:
    - torch.Tensor: The predicted values vector of size (n_samples, 1).
    """
    return torch.mm(X, w)

def squared_loss(y_hat, y):
    """
    Calculate the mean squared error loss.

    This function computes the mean squared error (MSE) between the predicted values (y_hat)
    and the actual values (y). It is used to evaluate the performance of the regression model.

    Parameters:
    - y_hat (torch.Tensor): Predicted values tensor, should be a 2D tensor with shape (n_samples, 1).
    - y (torch.Tensor): Actual values tensor, should be a 2D tensor with shape (n_samples, 1).

    Returns:
    - torch.Tensor: The mean squared error loss as a scalar tensor.
    """
    num_ = y.numel()  # Total number of elements in y
    sse = torch.sum((y_hat - y) ** 2)  # Sum of squared errors
    return sse / num_  # Mean squared error

def sgd(params, lr):
    """
    Perform a single step of stochastic gradient descent (SGD) on given parameters.

    This function updates the parameters based on the gradient of the loss function.
    It modifies the parameters in-place and resets their gradients after updating.

    Parameters:
    - params (torch.Tensor): Tensor containing the parameters to be updated.
    - lr (float): Learning rate, a scalar defining the step size during optimization.

    Returns:
    - None
    """
    params.data -= lr * params.grad  # Update the parameters
    params.grad.zero_()  # Reset gradients to zero after updating

def sigmoid(z):
    """
    Compute the sigmoid activation function.

    The sigmoid function is defined as 1 / (1 + exp(-z)), where `z` can be a number,
    a vector, or a matrix. This function is often used in logistic regression and
    neural networks to introduce nonlinearity.

    Parameters:
    - z (torch.Tensor): The input tensor.

    Returns:
    - torch.Tensor: The output tensor where the sigmoid function has been applied element-wise.
    """
    return 1 / (1 + torch.exp(-z))

def logistic(X, w):
    """
    Apply logistic regression model to input data.

    This function computes the predicted probabilities using a logistic regression model
    with parameters `w`. The logistic function is sigmoid(Xw), where X is the feature matrix
    and w is the coefficient vector.

    Parameters:
    - X (torch.Tensor): Feature matrix of size (n_samples, n_features).
    - w (torch.Tensor): Weight vector of size (n_features, 1).

    Returns:
    - torch.Tensor: Predicted probabilities for each sample (n_samples, 1).
    """
    return sigmoid(torch.mm(X, w))

def cal(sigma, p=0.5):
    """
    Calculate the class predictions from probability estimates.

    This function converts probability estimates into class predictions based on a threshold `p`.
    Values greater than or equal to `p` are classified as 1, and below `p` as 0.

    Parameters:
    - sigma (torch.Tensor): Probability estimates, typically from a logistic regression.
    - p (float, optional): The threshold for converting probabilities to class predictions. Default is 0.5.

    Returns:
    - torch.Tensor: Class predictions (0 or 1) based on the threshold.
    """
    return (sigma >= p).float()

def accuracy(sigma, y):
    """
    Calculate the accuracy of predictions.

    This function computes the accuracy as the percentage of correct predictions
    made by the model. It compares the predicted values (after thresholding) to the actual
    values `y`.

    Parameters:
    - sigma (torch.Tensor): Predicted probabilities before thresholding.
    - y (torch.Tensor): Actual labels (ground truth), should be of the same shape as the output of `cal`.

    Returns:
    - float: The accuracy of the predictions, a scalar value between 0 and 1.
    """
    predictions = cal(sigma).flatten()  # Compute binary predictions
    correct_predictions = predictions == y.flatten()  # Compare predictions with actual values
    return torch.mean(correct_predictions.float())  # Mean of correct predictions

def acc_zhat(zhat, y):
    """
    Compute logistic regression accuracy from the output of a linear model.

    This function applies the sigmoid activation function to the outputs of a linear equation (zhat),
    then computes the accuracy by comparing the thresholded sigmoid values to the actual labels.
    The sigmoid function maps the linear outputs to probabilities, and accuracy is computed as the
    proportion of correct predictions.

    Parameters:
    - zhat (torch.Tensor): The output tensor from a linear equation model. This tensor typically represents the logits.
    - y (torch.Tensor): Actual binary labels (0 or 1) for each example in the dataset.

    Returns:
    - float: The accuracy of predictions, represented as a float between 0 and 1.
    """
    sigma = sigmoid(zhat)  # Convert logits to probabilities using sigmoid
    return accuracy(sigma, y)  # Calculate and return the accuracy

def cross_entropy(sigma, y):
    """
    Compute the cross-entropy loss between predictions and actual values.

    The cross-entropy loss is a common loss function for classification tasks, especially
    with logistic models and neural networks. It measures the performance of a classification
    model whose output is a probability value between 0 and 1.

    Parameters:
    - sigma (torch.Tensor): Predicted probabilities, each element should be between 0 and 1.
    - y (torch.Tensor): Actual binary labels (0 or 1), must have the same shape as `sigma`.

    Returns:
    - torch.Tensor: The computed cross-entropy loss as a scalar.
    """
    loss = -(1 / y.numel()) * torch.sum((1 - y) * torch.log(1 - sigma) + y * torch.log(sigma))
    return loss

def softmax(X, w):
    """
    Apply the softmax function to the linear combinations of inputs and weights.

    The softmax function is used primarily in multi-class classification problems. It turns
    logits (the outputs of linear layers) into probabilities by taking the exponent of each output
    and then normalizing these values by dividing by the sum of all exponents.

    Parameters:
    - X (torch.Tensor): Input features matrix of size (n_samples, n_features).
    - w (torch.Tensor): Weight matrix of size (n_features, n_classes).

    Returns:
    - torch.Tensor: The softmax probabilities for each class, shape (n_samples, n_classes).
    """
    logits = torch.mm(X, w)
    exp_logits = torch.exp(logits)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    softmax_output = exp_logits / sum_exp
    return softmax_output

def m_cross_entropy(soft_z, y):
    """
    Calculate the mean cross-entropy loss for multi-class classification using softmax outputs.

    This function computes the cross-entropy loss, which is a measure of the difference between
    the true distribution (encoded in `y`) and the predicted distribution (`soft_z`). This is typically
    used when the outputs are probabilities obtained via softmax.

    Parameters:
    - soft_z (torch.Tensor): Softmax probabilities for each class. Each row corresponds to a single example
                             and should sum to 1. The shape is (n_samples, n_classes).
    - y (torch.Tensor): Actual labels (indices of the true class). Shape is (n_samples,).

    Returns:
    - torch.Tensor: The mean cross-entropy loss computed across all samples as a scalar.
    """
    y = y.long()  # Ensure labels are in long format for indexing
    prob_real = torch.gather(soft_z, 1, y.view(-1, 1))  # Gather the probabilities corresponding to the true labels
    return -(torch.log(prob_real).sum() / y.numel())  # Compute mean cross-entropy loss

def m_accuracy(soft_z, y):
    """
    Calculate the classification accuracy based on softmax probabilities.

    Accuracy is the proportion of true results (both true positives and true negatives) among the total number
    of cases examined. This function determines the predicted class as the one with the highest probability
    from softmax, then compares these predictions to the actual labels.

    Parameters:
    - soft_z (torch.Tensor): Softmax probabilities for each class, shape (n_samples, n_classes).
    - y (torch.Tensor): Actual labels, shape (n_samples,).

    Returns:
    - float: Accuracy as a float, representing the proportion of correctly predicted samples.
    """
    predictions = torch.argmax(soft_z, dim=1)  # Find the predicted classes (indices with max probability)
    correct_predictions = predictions.flatten() == y.flatten()  # Compare predictions to true labels
    return torch.mean(correct_predictions.float())  # Calculate mean accuracy



