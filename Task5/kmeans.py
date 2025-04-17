import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

"""
The algorithm goes by following the steps -->
1. Apply Kmeans clustering to all the points
2. If a cluster has less than min_threshold points consider them as noise
3. If a cluster has more than max_threshold points, recursive apply Kmeans to each cluster 
        until every cluster has less than 55 points
4. Now take any four random points in a given cluster and find average distance, do that for every cluster and take average of all.
        Let it be avgdist
5. If the cluster's average distance is more than N times avgdist consider them as noise
6. Merge clusters based on distance between centroids (smallest ones (based on number of points) first).
        A cluster always merges with cluster whose centroids is nearest.
7. This merging happens till the number of clusters is the given amount not more not less
"""

# I don't exactly know why some random warning appear. So, to make sure no warnings appear
warnings.filterwarnings("ignore")

###########################
# Utility Functions (Data Conversion)
###########################

def convert_to_float(coord):
    """
    Convert a coordinate string like "21.9161 N" or "77.6234 E"
    to a float value (negative for S/W).
    """

    # To check is a coordinate is proper or not, if not then just represent it as nan
    if pd.isna(coord):
        return np.nan
    
    # There is some coordinate in form _number_ N, to handle such cases
    # First convert the coord into a string for processing
    coord = str(coord)

    # Not, if it has any direction specifier in the last, then go into the block
    # Else just convert it to float, (if it can't in that case as well, just mark it as nan)
    if coord[-1] in ['N', 'E', 'S', 'W']:

        # The numerical value of the coordinate is everything expect the last character (N, E, S or W)
        value_str = coord[:-1]

        # In case it can be converted to a float do it, otherwise mark it as nan
        try:
            value = float(value_str)
        except ValueError:
            return np.nan
        
        # If the direction is South or West, take negative of the coordinate and return it (Similar to general coordinate plane)
        if coord[-1] in ['S', 'W']:
            return -value
        return value
    try:
        return float(coord)
    except ValueError:
        return np.nan

def read_state_data(file_path, state_name):
    """
    Read CSV file, convert coordinates, and filter rows by STATE name.
    Returns a dictionary containing a numpy array under key 'kmeansdata'.
    """

    # Read the csv file as string type in every column
    # If we can't read properly, return None
    try:
        df = pd.read_csv(file_path, dtype=str)
    except Exception as e:
        print("Error reading CSV file:", e)
        return None
    
    # After reading, convert every element under the latitude and longitude column using the convert_to_float function
    # Now, store all the data related to specific STATE (whose name is input) in stata_data
    # .lower() is used so that the STATE name input is case insensitive
    df['Latitude'] = df['Latitude'].apply(convert_to_float)
    df['Longitude'] = df['Longitude'].apply(convert_to_float)
    state_data = df[df['StateName'].str.lower() == state_name.lower()]  
    
    # In case there exists no such STATE, return None
    if state_data.empty:
        print(f"No records found for STATE: {state_name}")
        return None

    # After reading, drop the element whose value is Nan under the column Latitude and Longitude
    # Convert every element under the latitude and longitude column into numpy arrays of type float
    state_data = state_data.dropna(subset=['Latitude', 'Longitude'])
    latitudes = state_data['Latitude'].astype(float).values
    longitudes = state_data['Longitude'].astype(float).values

    # Store coordinates in the format [[lat1, long1], [lat2, long2], ...]
    kmeansdata = np.column_stack((latitudes, longitudes))

    # Return the coordinates
    return {'kmeansdata': kmeansdata}

###########################
# Clustering Functions
###########################

def basic_kmeans(data, num_clusters, iterations=100):
    """
    Basic K-means clustering using Euclidean distance.
    Returns (clusters, centroids, labels) where:
      clusters: list of arrays (points in each cluster)
      centroids: array of centroids
      labels: integer labels for each point.
    """

    # Get the number of rows and columns for the data
    rows, columns = data.shape

    # Create a new numpy array of dimensions number of cluster and columns (column == sizeof(coordinate)) to store centroids
    centroids = np.zeros((num_clusters, columns))

    # Initialize centroids uniformly between the max and the min value in the entire data for each column
    for j in range(columns):
        min_val, max_val = data[:, j].min(), data[:, j].max()
        centroids[:, j] = np.random.uniform(min_val, max_val, num_clusters)
    
    """
    The main kmeans algorithm --> 
    1. Calculate the distance between each point and the centroids
    2. A point belongs to cluster whose centroid is closest to the point
    3. Label that point i if it closest to ith centroid (label is the index of the centroid the point is closest to)
    4. Do the same thing for all the points
    5. Now update the centroids as the average coordinate of all the points belonging to a cluster
    6. Repeat the above steps N times
    7. Return the list of array of points in each cluster, list of centroids and integer label of each point
    """
    
    # Run this for iterations number of iterations
    for _ in range(iterations):

        # Calculating the distance of each point and centroids is done by np.linalg.norm
        # If point is [x0, y0] and centroid is [xc, yc] then point - centroid with axis = 1 gives [x0 - xc, y0 - yc]
        # np.linalg.norm calculates the norm of vector with elements (x0 - xc) and (y0 - yc)
        # np.argmin returns the index of the centroid whose norm turns out to be minimum
        # This is done for all the points in data and the labels are stored in numpy array called labels
        labels = np.array([np.argmin(np.linalg.norm(point - centroids, axis=1)) for point in data])

        # New_centroids numpy array acts as a temporary variable to do calculation for finding new centroids
        new_centroids = np.zeros((num_clusters, columns))

        # For each cluster do the following
        for k in range(num_clusters):
            # Check if any the cluster labelled k has any points or not, if it has then do the following
            if np.any(labels == k):
                # Get all points assigned to cluster k
                cluster_points = data[labels == k]
                # Compute average of coordinates (x and y)
                # Axis = 0 means compute the average down the column
                """
                For the following data set -->
                            col1, col2, col3
                row1  --   [[1,    0,    0],
                row2  --    [2,    3,    4],
                row3  --    [5,    2,    5]]
                .mean(axis = 0) means it calculates the average of (1,2,5), (0,3,2) and (0,4,5)
                .mean(axis = 1) means it calculates the average of (1,0,0), (2,3,4) and (5,2,5)
                """
                average_position = cluster_points.mean(axis=0)
                # Set the new calculated position as the new centroid
                new_centroids[k] = average_position
            # In case, there aren't any points, leave the centroid as it is
            else:
                new_centroids[k] = centroids[k]
        # Assign the new_centroids to the centroids
        centroids = new_centroids
    # After running the above sequence N times -->
    # Do the labelling again
    labels = np.array([np.argmin(np.linalg.norm(point - centroids, axis=1)) for point in data])
    # create clusters variable which stores an list of arrays of coordinates of points in each cluster
    # One array corresponds to one cluster
    clusters = [data[labels == k] for k in range(num_clusters)]

    # Return the clusters, centroids, and labels
    return clusters, centroids, labels

def recursive_kmeans_with_indices(data, indices, max_threshold=55, iterations=100):
    """
    Recursively apply K-means clustering to split clusters exceeding max_threshold.
    
    Returns a list of tuples in the form [(indices_1, data_1), (indices_2, data_2), …, (indices_M, data_M)]
    each (global_indices, cluster_data) represents a cluster that has <= max_threshold points.
    Global_indices are the indices of points in the original data which belong to the specific cluster
    Cluster_data is the list of coordinates of all the points in that cluster
    """

    # Assign rows as number of rows in data (data here is all the points)
    rows = data.shape[0]

    # In case the number of points in cluster is already less than max_threshold, return as it is
    if rows <= max_threshold:
        return [(indices, data)]
    
    # In case not, then the number of clusters it should be divided into is given by rows/max_threshold
    num_clusters = int(np.ceil(rows / max_threshold))

    # Run the kmeans algorithm on the data such that it gets divided into num_cluster number of clusters
    clusters, centroids, labels = basic_kmeans(data, num_clusters, iterations)

    # We have to return an array of tuples, where each tuple contains a list of indices of points in the original data which
    #          belong to a specific cluster, the array elements are in the form (global_indices, cluster_data)
    clusters_with_indices = []
    
    # After applying kmenas to the data, for each cluster we check if the number of points is > max_threshold or not
    # If it is, then we apply recursive kmeans again, else we will leave as it is
    for k in range(num_clusters):
        # Mask has all the indices of elements in label whose value is k
        mask = (labels == k)

        # All the elements whose label is k are stored in subdata
        # And their global indices of points in that subdata in subindices
        subdata = data[mask]
        subindices = indices[mask]

        # In case the number of elements in subdata is more than max_threshold then apply the function recursively
        if subdata.shape[0] > max_threshold:
            subclusters = recursive_kmeans_with_indices(subdata, subindices, max_threshold, iterations)
            # Add the subclusters to array clusters_with_indices
            clusters_with_indices.extend(subclusters)
        # If not then just extend the existing clusters_with_indices with the tuple (subindices, subdata)
        else:
            clusters_with_indices.append((subindices, subdata))
    
    # Return the clusters_with_indices array
    return clusters_with_indices

def filter_small_clusters(clusters_with_indices, min_threshold):
    """
    Filter out clusters with fewer than min_threshold points.
    Returns (valid_clusters, noise_indices) where noise_indices is an array of global indices.
    valid_clusters is an array whose content is similar to clusters_with_indices. It's elements are tuples (global_indices, cluster_points)
    """

    # Initialize the valid and noise array
    valid = []
    noise = []

    # For each array of global_indices and points in the cluster_with_indices array of tuples
    for idx_arr, points in clusters_with_indices:
        # If the number of points are less than min_threshold then consider them as noise
        # Add the global indices of those points in the noise array
        if len(points) < min_threshold:
            noise.extend(idx_arr)

        # Else, consider them as a valid cluster and extend the valid array with the tuple (global_indices of the points, collection of points)
        else:
            valid.append((idx_arr, points))
    
    # Return the valid clusters and the noise points
    return valid, np.array(noise)

def distance_noise_filter(clusters_with_indices, sample_size=4, macro=1.0):
    """
    For each cluster, sample exactly 4 points (if possible) and compute average pairwise distance.
    Compute overall mean dispersion. Clusters with dispersion greater than (macro * overall_mean)
    are marked as noise.
    Returns (filtered_clusters, noise_indices).
    """

    # cluster_dispersion contains the average distance of 4 points in each cluster according to the index of the cluster
    cluster_dispersion = []

    # For each cluster with more than sample_size points (all of them basically if it is less than in_threshold),
    #       sample sample_size points. Store those points in sample
    for idx_arr, points in clusters_with_indices:
        if points.shape[0] >= sample_size:
            # replace = False means same point can't be chosen more than once
            sample = points[np.random.choice(points.shape[0], sample_size, replace=False)]
        else:
            sample = points
        
        # Take the distance between each of those points
        dists = []
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                dists.append(np.linalg.norm(sample[i] - sample[j]))
        
        # Take mean of all those distances unless there is nothing in dists
        avg_dist = np.mean(dists) if dists else 0
        
        # Store that average distance in cluster_dispersion
        cluster_dispersion.append(avg_dist)
    
    # Take the mean of all such distances unless there is nothing in cluster_dispersion
    overall_mean_disp = np.mean(cluster_dispersion) if cluster_dispersion else 0

    # All the cluster's whose sample average is greater than N times the average are considered as noise
    # All others are valid clusters
    valid = []
    noise = []
    for i, (idx_arr, points) in enumerate(clusters_with_indices):
        if cluster_dispersion[i] > (macro * overall_mean_disp):
            noise.extend(idx_arr)
        else:
            valid.append((idx_arr, points))

    # Return the valid clusters (In the same format as before) and noise (noise points' global indices)
    return valid, np.array(noise)

def merge_clusters_until_target(clusters_with_indices, target_num):
    """
    Iteratively merge clusters based on centroid proximity until the number of clusters equals target_num.
    Always merge the smallest cluster with its nearest neighbor.
    """

    # In the function it self, define a function which finds the centroid of a set of points
    def compute_centroid(points):
        return np.mean(points, axis=0) if points.shape[0] > 0 else None

    # Copy the clusters_with_indices array
    clusters = clusters_with_indices.copy()

    # While the number of clusters is more than target number do the following
    while len(clusters) > target_num:

        # Each element in clusters is of the form (global_indices_of_points, collection_of_points_in_cluster)
        # len(element[0]) in clusters is the length of the specific cluster
        # Here clusters are sorted based of length of the element[0]
        # That is clusters are sorted based of number of points in the cluster 
        clusters = sorted(clusters, key=lambda x: len(x[0]))

        # Indices and collections of points in the smallest cluster is saved
        smallest_indices, smallest_points = clusters[0]

        # Centroid of the smallest cluster is calculated
        centroid_small = compute_centroid(smallest_points)
        # Now we have to find which cluster's centroid is nearest to the smallest cluster's centroid and merge smallest cluster with it

        # Initialize minimum distance between smallest cluster's centroid and other cluster's centroid to be infinity
        # Initialize the index of the cluster to None as well
        min_dist = float('inf')
        merge_index = None

        # Loop throgh all the cluster's centroids to check which cluster is the nearest
        for i in range(1, len(clusters)):
            curr_centroid = compute_centroid(clusters[i][1])
            columns = np.linalg.norm(centroid_small - curr_centroid)
            # In case some cluster's distance is less than min_dist, assign it to be the minimum distance
            #       and assign merge_index to be index of that specific cluster
            if columns < min_dist:
                min_dist = columns
                merge_index = i
        
        # idx1 is the array of global indices of points in the smallest cluster, pts1 is the collection of points in smallest cluster
        # idx2 is the array of global indices of points in the cluster nearest to smallest cluster
        # pts2 is the collection of points in the cluster nearest to smallest cluster
        idx1, pts1 = clusters[0]
        idx2, pts2 = clusters[merge_index]

        # Combine both idx and idx2 and store them in merged_indices
        # Combine pts1 and pts2 and store them in merged_points
        merged_indices = np.concatenate((idx1, idx2))
        merged_points = np.vstack((pts1, pts2))

        # Create a new collection of clusters without smallest cluster and the nearest cluster
        # Add the newly created cluster to that collection in the cluster_with_indices form 
        new_clusters = [clusters[i] for i in range(len(clusters)) if i not in [0, merge_index]]
        new_clusters.append((merged_indices, merged_points))

        # assign clusters as the new collection and do this till num_of clusters is less than or equal to target_num
        clusters = new_clusters
    
    # Return clusters after the merging process
    return clusters

###########################
# Ensemble (Consensus) Functions
###########################

def run_pipeline(data, global_indices, MAX_THRESHOLD, MIN_THRESHOLD, KMEANS_ITER, MACRO_THRESHOLD):
    """
    Runs steps 1-6 of the algorithm on the provided data.
    Returns a label array (length = # of points) with noise as -1.
    """

    # Call all the required functions one by one
    clusters_all = recursive_kmeans_with_indices(data, global_indices,
                                                 max_threshold=MAX_THRESHOLD,
                                                 iterations=KMEANS_ITER)
    valid_clusters, noise_indices = filter_small_clusters(clusters_all, MIN_THRESHOLD)
    valid_clusters, noise_indices2 = distance_noise_filter(valid_clusters, sample_size=4, macro=MACRO_THRESHOLD)
    # By this point, we have all the proper clusters in valid_clusters

    # We have to return final label of all the points which are not noise with respect to global index
    # Initialize final labels  as -1 (indicating noise)
    final_labels = np.full(data.shape[0], -1, dtype=int)

    # In all the valid_clusters, we will go through the global indices of points in the cluster and assign proper label to them
    for label, (indices_arr, _) in enumerate(valid_clusters):

        # If a cluster's index is N, then we will assign N to all the elements in final label with indices same as global indices of points in the cluster 
        final_labels[indices_arr] = label
    
    # Return the final_labels array
    return final_labels

def clusters_from_labels(labels, data):
    """
    Groups points by label and returns a list of clusters as (indices, data) pairs.
    """
    # Initialize clusters
    clusters = {}

    # For every label in labels
    for i, lab in enumerate(labels):
        # if label is -1, consider it as noise and skip it
        if lab == -1:
            continue
        
        # Else, If lab is already a key in the clusters dictionary, append i to its list.
        # If lab is not yet a key, create a new list for it first, then append i.
        # If it key or not if checked byy setdefault
        clusters.setdefault(lab, []).append(i)
    
    # Clusters which we have to return
    cluster_list = []

    # Iterate over each cluster label and its corresponding list of indices
    for lab, idx_list in clusters.items():
    
        # Convert the list of indices to a sorted NumPy array
        idx_array = np.array(sorted(idx_list))
    
        # Use the indices to extract the corresponding data points from the dataset,
        # and append the pair (indices, data points) to the cluster list
        cluster_list.append((idx_array, data[idx_array]))

    # Return the final list of clusters, each represented as (indices, data points)
    return cluster_list


###########################
# Main Processing Pipeline (Ensemble)
###########################

# Parameters (adjust as needed)
FILE_PATH = 'clustering_data.csv'
MAX_THRESHOLD = 55           # Maximum number of points per cluster (Maximum pincodes in a district of Telangana state is 55)
MIN_THRESHOLD = 10           # Minimum number of points per cluster; otherwise, considered noise (Minimum is 10)
TARGET_CLUSTERS = 33         # Desired final number of clusters (Number of districts are 33)
KMEANS_ITER = 100            # Iterations for basic k-means
MACRO_THRESHOLD = 12.0       # Macro multiplier for average distance threshold in noise filtering
STATE = 'Telangana'          # Name of the state
 
# Read the state data for the given state name
# If no such state exists, return None
state_dict = read_state_data(FILE_PATH, STATE)
if state_dict is not None and 'kmeansdata' in state_dict:
    data = state_dict['kmeansdata']
else:
    data = None

# If there is no data, exit
if data is None:
    exit("No data to process.")

# Create the global_indices based on ascending order of indices of points in data
# This is to ignore all the data which is in between the data we want
global_indices = np.arange(data.shape[0])

# Run the pipeline (steps 1–6)
labels_run1 = run_pipeline(data, global_indices, MAX_THRESHOLD, MIN_THRESHOLD, KMEANS_ITER, MACRO_THRESHOLD)

# Convert consensus labels to clusters
consensus_clusters = clusters_from_labels(labels_run1, data)

# Merge clusters based on centroid proximity until exactly TARGET_CLUSTERS remain
if len(consensus_clusters) > TARGET_CLUSTERS:
    merged_clusters = merge_clusters_until_target(consensus_clusters, TARGET_CLUSTERS)
else:
    merged_clusters = consensus_clusters

# Build final labels from merged clusters; assign noise (-1) to points not present.
final_labels = -1 * np.ones(data.shape[0], dtype=int)
for label, (indices_arr, _) in enumerate(merged_clusters):
    final_labels[indices_arr] = label

###########################
# Plotting Results
###########################


# Compute the sorted array of unique cluster labels (including –1 for noise)
plt.figure(figsize=(8, 6))
unique_labels = np.unique(final_labels) 


# Generate a distinct colormap entry for each label by sampling the 'Spectral' colormap 
# evenly between 0 and 1.  
# ‘Spectral’ is a diverging palette ideal for categorical distinctions. (apparently, idk tbh)  
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Skip the noise label (–1) so that those points are not plotted
        continue

    mask = (final_labels == k)
    # Build a boolean mask selecting all points assigned to cluster k

    plt.scatter(
        data[mask, 1],         # x-coordinates: longitude values  
        data[mask, 0],         # y-coordinates: latitude values  
        s=30,                  # marker size in points^2  
        alpha=0.6,             # semi-transparent markers  
        c=[col],               # use the colormap color for cluster k  
        label=f"Cluster {k}"   # add a legend entry for this cluster
    )

plt.title(f"Final Consensus Clusters (Target: {TARGET_CLUSTERS})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# Sometimes the results are wonky, need to find a way to run the same pipeline 4-5 times and take the best result
# Have to wait for some 15s to get the output