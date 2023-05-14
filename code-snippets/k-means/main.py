import math
import random

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return math.sqrt(sum([(i - j) ** 2 for i, j in zip(a, b)]))

# Function to calculate mean
def mean(points):
    length = len(points)
    return tuple(sum(p[i] for p in points) / length for i in range(len(points[0])))

# K-means function
def k_means(data, k, max_iters=100):
    # Initialize centroids randomly from the data points
    centroids = random.sample(data, k)

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = [min(range(k), key=lambda i: euclidean_distance(point, centroids[i])) for point in data]

        # Calculate new centroids as the mean of the assigned points
        new_centroids = [mean([point for point, label in zip(data, labels) if label == i]) for i in range(k)]

        # Check for convergence
        if centroids == new_centroids:
            break

        centroids = new_centroids

    return labels, centroids

# Test data
data = [(1, 1), (1.5, 2), (3, 3), (5, 7), (3.5, 5), (4.5, 5), (3.5, 4.5)]

# Apply the k-means algorithm
labels, centroids = k_means(data, 2)

print("Labels:", labels)
print("Centroids:", centroids)

