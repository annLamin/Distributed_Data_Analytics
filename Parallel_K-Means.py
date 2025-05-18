from mpi4py import MPI
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Question 1
data = pd.read_csv('cluster_data.csv', header=0, index_col=0)
print(data.head(10))

# scatter_data = data[['x', 'y']].values
# plt.scatter(scatter_data[:, 0], scatter_data[:, 1], s=10)
# plt.show()



# Set up logging
logging.basicConfig(format='%(name)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(f'Process {rank}')

# Parameters
K = 3  # Number of clusters
MAX_ITER = 4
TOLERANCE = 1e-4

# Root process loads and prepares the data
if rank == 0:
    # Load your CSV dataset (modify path as needed)
    df = pd.read_csv('cluster_data.csv', header=0, index_col=0)
    
    # Convert to numpy array of integers
    data = df.values.astype('float64')
    n_samples, n_features = data.shape
    logger.info(f"Loaded dataset with {n_samples} samples and {n_features} features")
else:
    data = None
    n_samples = None
    n_features = None

# Broadcast dataset dimensions to all processes
n_samples = comm.bcast(n_samples, root=0)
n_features = comm.bcast(n_features, root=0)

# Calculate how to split the data
local_size = n_samples // size
remainder = n_samples % size

# Prepare counts and displacements for Scatterv
counts = [local_size + 1 if i < remainder else local_size for i in range(size)]
displacements = [sum(counts[:i]) for i in range(size)]

# Create buffer for local data
local_data = np.empty((counts[rank], n_features), dtype='float64')

# Scatter the data
comm.Scatterv([data, counts, displacements, MPI.DOUBLE], local_data, root=0)
logger.info(f"Received {local_data.shape[0]} samples")

# Initialize centroids (root selects initial centroids)
if rank == 0:
    # Randomly select K data points as initial centroids
    centroids = data[np.random.choice(n_samples, K, replace=False)]
    logger.info("Initial centroids selected:")
    for i, c in enumerate(centroids):
        logger.info(f"Cluster {i}: {c}")
else:
    centroids = np.empty((K, n_features), dtype='float64')

# Broadcast initial centroids to all processes
comm.Bcast(centroids, root=0)

# K-Means main loop
for iteration in range(MAX_ITER):
    # Calculate distances to centroids for local data
    distances = np.zeros((local_data.shape[0], K))
    for i in range(K):
        distances[:, i] = np.linalg.norm(local_data - centroids[i], axis=1)
    
    # Assign each point to nearest centroid
    local_assignments = np.argmin(distances, axis=1)
    
    # Gather all assignments to root
    if rank == 0:
        all_assignments = np.empty(n_samples, dtype='float64')
    else:
        all_assignments = None
    
    comm.Gatherv(local_assignments, 
                [all_assignments, counts, displacements, MPI.Double], 
                root=0)
    
    # Gather all data points to root (for centroid calculation)
    if rank == 0:
        all_data = np.empty((n_samples, n_features), dtype='float64')
    else:
        all_data = None
    
    comm.Gatherv(local_data, 
                [all_data, 
                 [c * n_features for c in counts],
                 [d * n_features for d in displacements],
                 MPI.INT], 
                root=0)
    
    # Root calculates new centroids
    if rank == 0:
        new_centroids = np.zeros_like(centroids)
        cluster_counts = np.zeros(K)
        
        # Sum all points in each cluster
        for i in range(n_samples):
            cluster = all_assignments[i]
            new_centroids[cluster] += all_data[i]
            cluster_counts[cluster] += 1
        
        # Compute new centroids (using integer division)
        for i in range(K):
            if cluster_counts[i] > 0:
                new_centroids[i] = (new_centroids[i] / cluster_counts[i]).astype('float64')
            else:
                # Keep previous centroid if cluster is empty
                new_centroids[i] = centroids[i]
        
        # Check for convergence
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        logger.info(f"Iteration {iteration}: Shift = {centroid_shift}")
        
        if centroid_shift < TOLERANCE:
            logger.info("Converged!")
            converged = True
        else:
            converged = False
            centroids = new_centroids.copy()
    else:
        converged = False
    
    # Broadcast convergence status and new centroids
    converged = comm.bcast(converged, root=0)
    comm.Bcast(centroids, root=0)
    
    if converged:
        break

# Final results (root process only)
if rank == 0:
    logger.info("\nFinal Results:")
    for i in range(K):
        cluster_points = all_data[all_assignments == i]
        logger.info(f"Cluster {i} (Centroid: {centroids[i]}): {len(cluster_points)} points")
    
    # Plotting the clusters
    
    plt.figure(figsize=(10, 6))
    
    # Plot each cluster with different colors
    colors = ['red', 'blue', 'green'][:K]
    
    for i in range(K):
        cluster_mask = (all_assignments == i)
        cluster_points = all_data[cluster_mask]
        
        # Plot points
        plt.scatter(
            cluster_points[:, 0],  # First feature on x-axis
            cluster_points[:, 1],  # Second feature on y-axis
            c=colors[i],
            label=f'Cluster {i}',
            alpha=0.6
        )
        
        # Plot centroid
        plt.scatter(
            centroids[i, 0],
            centroids[i, 1],
            c='black',
            marker='x',
            s=100,
            linewidths=2
        )
        
        logger.info(f"Cluster {i} (Centroid: {centroids[i]}): {len(cluster_points)} points")

    # Add labels and legend
    plt.title('KMeans Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Save and show
    plt.savefig('kmeans_clusters.png')
    plt.show()
    
    
    # Calculate WCSS (Within-Cluster Sum of Squares)
    wcss = 0
    for i in range(n_samples):
        cluster = all_assignments[i]
        wcss += np.linalg.norm(all_data[i] - centroids[cluster]) ** 2
    logger.info(f"Final WCSS: {wcss}")