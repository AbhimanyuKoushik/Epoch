# Importing required Libraries
import numpy as np
import tkinter as tk
from tkinter import ttk


# For implementing Decision Tree Algorithm we have to go through the following steps -->
# 1. Encode Data, that is, get Data into numerical form. For this case, it is changing labels to integers and
#       # separating features and labels into a matrix and vector respectively
# 2. Creating a Base node which includes all the processed data. We will use this base node to grow the tree
        # This Base node is called Root node
# 3. We will try to find the best way to split this data into two child nodes. This splitting is the major step
        # The most common methods to split the tree is using Gini impurity or Entropy
        # Here we will use Both
# 4. Recursively using this algorithm until we get a stopping criteria
        # Here the stopping criterias are maximum depth (size) of the tree, minimum number of samples in a node
        # and threshold Impurity or Entropy

# The Given Data Set
data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
]

# Defining the Labels and reverse mapping as well
label_dictionary = {0:'Wine', 1:'Beer', 2:'Whiskey'}
encoded_mapping = {key: item for item, key in label_dictionary.items()}

# Getting the data where the Drink's Label is used instead of it's name
labeled_data = [[row[0], row[1], row[2], encoded_mapping[row[3]]] for row in data]

# The collection of all the attributes of the Drink in a matrix is FeatureMatriz
# The Label (or the Drink) corresponding to each row of the Matrix is Label Vector
feature_matrix_X = np.array([[row[0], row[1], row[2]] for row in labeled_data], dtype = float)
label_vector_Y = np.array([row[3] for row in labeled_data], dtype = float)

# Functions calculates Gini impurity for a given set of labels
# Labelled set will be in the following way -->
#       gini impurity = 1 - sum(p_i**2)
def gini_impurity(labels):

    # It stores a list of all the unique labels in first list, and the frequency of each element in first list in the second list
    # example, if labels = [1,1,3,2,1,2,2,6,1] then
    # unique_labels = [1,2,3,6] and count_of_labels = [4,3,1,1] 
    unique_labels, count_of_labels = np.unique(labels, return_counts = True)

    # probabilites is a list which contains the probability of each unique element at same index
    # len(<list>) gives length of list while <nparray>.size gives length of the numpy array
    probabilites = count_of_labels/labels.size

    # If we want to calculate entropy we can use the expression below
    # return -np.sum(probabilities * np.log2(probabilities))

    # Return 1 - sum of square of all the probabilities
    return 1 - np.sum(probabilites ** 2)

# Creating a Node class with attributes -->
# 1. feature_index (The feature on which decision is made and child nodes are created)
# 2. threshold (The lowest Gini impurity or Entropy is where we classify the type of object,
        # threshold is the line which differentiates two objects)
# 3. left (One of the child)
# 4. right (The other child)
# 5. value (If the test point belongs to this (only if there are no left and right), what value should be returned,
        # basicaally, predicted label in that leaf node)
class Node:
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Creating the DecisionTree class
class DecisionTree:
    # Creating a constructor for the DecisionTree
    # By default root is None, min_samples split is 2 and max_depth is 10
    def __init__(self, min_samples_split = 2, max_depth = 10, root = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = root
    
    # A function which loads the data.
    # Initializing the root node with feature_matrix and label_vector with depth 0 useing grow tree function
    def fit(self, feature_matrix, label_vector):
        self.root = self.grow_tree(feature_matrix, label_vector, depth = 0)

    # Grow tree function recursively builds the Decision tree
    def grow_tree(self, feature_matrix, label_vector, depth = 0):
        
        # Storing number of data points in training data 
        number_of_samples = feature_matrix.shape[0]
        number_of_labels = len(np.unique(label_vector))
        
        # Check stopping criteria, if not then -->
        # Find the best split -->
        # Make child nodes

        # Check stopping criteria -->
        # 1. Stop is all the labels are same
        # 2. Stop if max_depth is reached
        # 3. Number of samples is below required, less than min_sample_split
        if np.unique(label_vector).size == 1 or number_of_samples < self.min_samples_split or depth >= self.max_depth:
            # If any of the above criteria is reached then the node is a leaf node (no children) with
                # the elements in it belong to the class which has majority of elements in that specific node
            # For example is the node contains labels [1,1,2,3,1,2,3,4,1], since it contains label one the most
                # if any of the test case reaches that node, it is assigned the label 1
            # We will store the number of unique values in the list counts
            values, counts = np.unique(label_vector, return_counts=True)

            # The label with majority of the number of element (max count) is the final label which has to be assigned
            # majority class is the label which has maximum number of elements in the leaf node
            majority_class = values[np.argmax(counts)]

            # We will return the node which is labelled the majority class for the leaf node
            return Node(value=majority_class)
        
        # In case we won't reach any stopping criteria we will split the node with the function best_split (defined later)
        # The output of the best_split function will be --> 
        # 1. feature index (the feature based on which the function is getting split)
        # 2. threshold (the threshold for the feature used to separate elements)
        # 3. impurity (the impurity or entropy we get for particular node split)
        feature_index, threshold, impurity = self.best_split(feature_matrix, label_vector)

        # In case the feature_index is none, we will consider it as a leaf node
        # and we will declare the label in which majority elements belong is the label we will be giving to test node 
        if feature_index == None:
            values, counts = np.unique(label_vector, return_counts=True)
            majority_class = values[np.argmax(counts)]
        
        # In all the other cases, we will go through each element in the node and classify it based on the feature and threshold
        # We will iterate through all the elements using np.where
        # left_feature and left_label are list of the particular points whose feature parameter is below the threshold value
            # and the labels corresponding the point
        # If it is above the threshold the point will in go the right_feature and the corresponding label to right_label
        left_feature, left_label = [], []
        right_feature, right_label = [], []

        # zip(A,B) basically combines the A, B elements into a tuple like datatype called zip
        # for example zip([1, 0], [0, 1]) to ((1, 0),(0, 1)), we can only enumerate through zip objects, they are not subscriptable
        for xi, yi in zip(feature_matrix, label_vector):
            # We will put element in left if it below the threshold, other wise in right
            if xi[feature_index] < threshold:
                left_feature.append(xi)
                left_label.append(yi)
            else:
                right_feature.append(xi)
                right_label.append(yi)
        
        # Now recursively call the function on both left and right
        # The inputs should be left_feature and left_label instead of feature_matrix and label_vector as only they are part of left
        # Same with right
        # And the dept should be increased by 1 as we traversed through one layer
        left_child = self.grow_tree(np.array(left_feature), np.array(left_label), depth + 1)
        right_child = self.grow_tree(np.array(right_feature), np.array(right_label), depth + 1)

        # Finally we will return root node as the output with right parameters
        return Node(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child)

    def best_split(self, feature_matrix, label_vector):
        """
        Find the best feature index and threshold for splitting the dataset
        using the average of the Gini impurity of the two splits.
        
        Returns:
            best_feature_index (int): Index of the best feature. (Index of column in feature matrix)
            best_threshold (float): Threshold value for the best split.
            best_impurity (float): Average Gini impurity value (unweighted) of the best split.
        """

        # matrix.shape outputs number_of_rows, number_of_columns
        number_of_points, number_of_features = feature_matrix.shape

        # Initializing the feature_index, threshold, and least impurity to be chosen
        best_feature_index = None
        best_threshold = None
        best_impurity = np.inf

        # Loop through every feature and then every point in the feature to find the proper threshold and feature_index
        for feature_index in range(number_of_features):
            # Feature value contains all the values of a particular feature
            feature_values = feature_matrix[:, feature_index]

            # We will consider all the unique values of the feature_values
            unique_values = np.unique(feature_values)

            # Consider midpoints between consecutive unique values as candidate thresholds.
            for i in range(1, unique_values.size):
                # Since we have to iterate through all possible intermediate points as candidate threshold it doesn't matter
                    # how you, might as well go linearly
                threshold = (unique_values[i - 1] + unique_values[i]) / 2.0

                # Split data into left and right branches.
                # The left_make would a list of 1s and 0s, 1 means the value is in left mask, 0 means its not
                left_mask = feature_values < threshold
                # Whatever is there in left mask can't be in right and vice versa, hence it will be not of each element in left mask
                right_mask = ~left_mask

                # Skip splits that lead to an empty branch.
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate normal Gini impurity for both sides.
                impurity_left = gini_impurity(label_vector[left_mask])
                impurity_right = gini_impurity(label_vector[right_mask])

                # Find the average of gini impurity (sum is also fine)
                # Apparently there is something called wieghted gini which considers number of labels in each node as well
                # weighted_gini = (no.left labels / total labels) * imp_left + (no.right labels / total labels) * imp_right
                avg_impurity = (impurity_left + impurity_right) / 2.0
                
                # If we find a lower impurity than exisiting one we will replace
                if avg_impurity < best_impurity:
                    best_impurity = avg_impurity
                    best_feature_index = feature_index
                    best_threshold = threshold

        # We will return the index of the best_feature and threshold where to split and the impurity
        return best_feature_index, best_threshold, best_impurity

    def predict(self, X):
        """
        Predict class labels for input samples X.
        """
        return np.array([self.predict_sample(x, self.root) for x in X])

    def predict_sample(self, x, node):
        """
        Recursively traverse the tree to predict the class label for a single sample.
        """
        # If the node is a leaf node (hence it has the value), then return its value
        if node.value is not None:
            return node.value
        
        # Else, check based on what feature the node is getting split and push the predicted value in to left or right
        # For the node's feature, if the test sample is less than threshold, push it into left and predict it
        # Else push it into right and predict it
        if x[node.feature_index] < node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)


# Instantiate and train the decision tree classifier.
tree = DecisionTree(max_depth=3, min_samples_split=2)
tree.fit(feature_matrix_X, label_vector_Y)

# Define test data.
test_data = np.array([
    [6.0, 2.1, 0],
    [39.0, 0.05, 1],
    [13.0, 1.3, 1]
])

# Generate predictions.
predictions = tree.predict(test_data)

# Convert numeric predictions to label strings using label_dictionary.
predicted_labels = [label_dictionary[pred] for pred in predictions]

print("Predictions for test data:")
for idx, sample in enumerate(test_data):
    print("Input:[", end="")
    for i in range(len(sample)):
        print(f"{sample[i]}", end=", " if i < len(sample) - 1 else "")
    print("]", end="")
    print(f" --> Predicted: {predicted_labels[idx]}")