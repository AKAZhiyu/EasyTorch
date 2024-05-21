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






