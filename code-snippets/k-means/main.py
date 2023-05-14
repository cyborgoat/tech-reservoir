import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def k_means(X, centroids, max_iters=100):
    k = centroids.shape[0]
    for _ in range(max_iters):
        # Step 2: Assign each data point to the nearest centroid
        labels = np.argmin([euclidean_distance(X, c) for c in centroids], axis=0)

        # Step 3: Update the centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# Generate example data
np.random.seed(0)
X = np.vstack([np.random.normal(loc=i, scale=1.0, size=(100, 2)) for i in range(1, 4)])

# Define initial centroids
centroids = np.array([[1, 1], [2, 2], [3, 3]])

# Apply the k-means algorithm
labels, final_centroids = k_means(X, centroids)

# Display the results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='x')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
