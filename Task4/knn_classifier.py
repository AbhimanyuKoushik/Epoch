# Importing required Libraries
import numpy as np

# Given Data
data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]

# Defining the Labels
label_dictionary = {0:'Apple', 1:'Banana', 2:'Orange'}

# Function to change the name of the Fruit to its Label
def label_fruit(fruit_name):
    for label in label_dictionary:
        if label_dictionary[label] == fruit_name:
            return label

# Getting the data where the Fruit's Label is used instead of it's name
labeled_data = [[row[0], row[1], row[2], label_fruit(row[3])] for row in data]

# The collection of all the attributes of the Fruit in a matrix is FeatureMatriz
# The Label (or the Fruit) corresponding to each row of the Matrix is Label Vector
feature_matrix_X = [[row[0], row[1], row[2]] for row in labeled_data]
label_vector_Y = [row[3] for row in labeled_data]

# Function to Calculate Distance between two points in Nth Dimension
def distance_between_points(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return np.sqrt(distance)

# Function for finding minkowski distance (put p = 1 for manhattan)
def distance_between_points_mink(point1, point2, p):
    distance = 0
    for i in range(len(point1)):
        distance += abs(point1[i] - point2[i]) ** p
    return distance ** (1/p)

# Defining the K-Nearest Neighbour classifier
# It is a simple algorithm based on the following steps:
# --> Get the Training Data
# --> To find which category an input point belongs to, find the distance with all the points in Training set
# --> Sort them
# --> Find the k points which are nearest to the point
# --> The type of element which has maximum number of nearest points in the type which input point belongs to
class KNN:
    # Defining the constructor
    def __init__(self, k = 3):
        # By default k is taken as 3
        self.k = k
    
    # Function to take the Data in form Feature Matrix and Label Vector
    def fit(self, feature_matrix, label_vector):
        self.feature_matrix = feature_matrix
        self.label_vector = label_vector

    # Function which predicts what Group a single element belongs to
    def predict_one(self, input_point):
        self.input_point = input_point

        # Finding the Distance between the input point and all the points
        # distance and Labels is a n x 2 vector (for n points)
        # first column is distance between points and input
        # Second column is the Label corresponding to points
        distances_with_labels = [[distance_between_points(self.feature_matrix[i], input_point), self.label_vector[i]]
                                 for i in range(len(self.feature_matrix))]
        
        # Sorting based on the Distance but keeping the labels intact
        # For example if the distances_with_labels was the set:
        #   [[5, 1], [3, 2], [2, 0], [4, 2]] then the sorted one looks like
        #   [[2, 0], [3, 2], [4, 2], [5, 1]] basically sorting wrt only distance
        distances_with_labels.sort(key = lambda x: x[0])

        # A List to check how many points belong to each type in the k nearest points
        label_frequency = [0] * 3 # Three because we have only three Labels

        # If an element in the first category in first k elements in sorted distance we will increase its count
        for pair in distances_with_labels[0:self.k]:
            label_frequency[pair[1]] += 1
            # For weighted KNN, we add weighted based on inverse of the distance, the closer the better
            # label_frequency[pair[1]] += 1 / distance --> Used for weighted KNN
        
        # The Category in which maximum elements belong to is taken
        majority_count = max(label_frequency)

        # Checking what the label corresponds to through dictionary defined at the start by iterating through all types
        best_label = -1
        for i in range(len(label_frequency)):
            if (label_frequency[i] == majority_count):
                best_label = i 
                break
        # Returns the Label
        return label_dictionary[best_label]

    # Function which does the same for a group of test points
    def predict(self, test_set):
        self.test_set = test_set
        # Creating a new List so that we don't change the original Training set
        predictions = test_set
        # Running the function to each element and storing it in the Label list
        predictions = [self.predict_one(predictions[i]) for i in range(len(test_set))]
        # Returning the List
        return predictions

    # Function to check accuracy when a test_set along with their true_labels are provided
    def check_accuracy(self, test_set, true_labels):
        self.true_labels = true_labels
        self.test_set = test_set
        # First predict the labels using predict function
        predicted_set = self.predict(test_set)

        # Now check the number of correct predictions, num_of_crct-prd / total_num_of_prd is accuracy
        # Initailize num_of_crct_prd to 0
        number_of_correct_predictions = 0

        # Iterate through each element in true label and check if its matching with predicted label
        # If yes, increase number of correct predictions by 1
        for elem in range(len(test_set)):
            if predicted_set[elem] == true_labels[elem]:
                number_of_correct_predictions+=1
        
        # Return the num_of_crct_prd / total_num_of_prd
        return (number_of_correct_predictions/len(true_set))
    
    # Function which return Z-score normalized data set
    def zscore_normalize(self, feature_matrix):
        # A Z-score normalized matrix is a Feature matrix where each element is calculated by the formula -->
        # zsfeature[i] = (feature[i] - mean_of_feature)/standard_deviation_of_feature
        self.feature_matrix = feature_matrix

        # Converting the feature matrix into numpy array for easy mean, std calculation
        feature_matrix = np.array(feature_matrix, dtype = float)

        # Finding the mean of every column (extracted by axis = 0) and storing in mean_features
        mean_features = feature_matrix.mean(axis = 0)

        # Finding the standard deviation of every column (extracted by axis = 0) and storing in standarddev_of_features
        standarddev_of_features = feature_matrix.std(axis = 0)

        # Finding normalized matrix and returning it
        normalized_matrix = (feature_matrix - mean_features)/standarddev_of_features
        return normalized_matrix

    # Function which return minmax normalized data set
    def minmax_normalize(self, feature_matrix):
        # A minmax normalized matrix is a Feature matrix where each element is calculated by the formula -->
        # mmfeature[i] = (feature[i] - min_of_feature)/(max_of_feature - min_of_feature)
        self.feature_matrix = feature_matrix

        # Converting the feature matrix into numpy array
        feature_matrix = np.array(feature_matrix, dtype = float)

        # Finding the min and max of every column and storing in minimum_features and maximum_features
        minimum_features = feature_matrix.min(axis = 0)
        maximum_features = feature_matrix.max(axis = 0)

        # Finding normalized matrix and returning it
        normalized_matrix = (feature_matrix - minimum_features) / maximum_features - minimum_features
        return normalized_matrix

def traintest_split(percentage_to_be_splitted, data):
    if (percentage_to_be_splitted > 100) or (percentage_to_be_splitted < 0):
        return None
    else:
        number_of_samples = len(data)
        number_of_elements_in_train = int(number_of_samples * (percentage_to_be_splitted / 100))
        number_of_elements_in_test = number_of_samples - number_of_elements_in_train
        
        train_data = [None] * number_of_elements_in_train
        test_data = [None] * number_of_elements_in_test
        
        # Fill the training set with the first part of the data
        for i in range(number_of_elements_in_train):
            train_data[i] = data[i]
        
        # Fill the test set with the rest of the data
        for j in range(number_of_elements_in_test):
            test_data[j] = data[number_of_elements_in_train + j]
        
        return train_data, test_data

test_data = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
])


myKNN = KNN()
myKNN.fit(feature_matrix_X, label_vector_Y)
predicted_labels = myKNN.predict(test_data)

print("Predictions for test data:")
for idx, sample in enumerate(test_data):
    print("Input:[", end="")
    for i in range(len(sample)):
        print(f"{sample[i]}", end=", " if i < len(sample) - 1 else "")
    print("]", end="")
    print(f" --> Predicted: {predicted_labels[idx]}")
