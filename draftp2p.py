from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def load_data(file_path='cluster_data.csv'):
    return np.load(file_path)

def initialize_centroids(data, k):
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)

def compute_local_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    counts = np.zeros(k)
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
            counts[i] = len(cluster_points)
    return centroids, counts

def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(8, 6))
    for i in range(centroids.shape[0]):
        points = data[labels == i]
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X', label='Centroids')
    plt.legend()
    plt.title('Distributed K-Means Clustering')
    plt.show()

def main():
    k = 4
    max_iters = 100
    tol = 1e-4

    # Load and distribute data
    if rank == 0:
        full_data = load_data()
        n_samples = full_data.shape[0]
        split_sizes = [n_samples // size + (1 if i < n_samples % size else 0) for i in range(size)]
        displacements = [sum(split_sizes[:i]) for i in range(size)]
    else:
        full_data = None
        split_sizes = None
        displacements = None

    split_sizes = comm.bcast(split_sizes, root=0)
    displacements = comm.bcast(displacements, root=0)
    local_data = np.zeros((split_sizes[rank], 2))
    comm.Scatterv([full_data, 
                   tuple(s * 2 for s in split_sizes), 
                   tuple(d * 2 for d in displacements), 
                   MPI.DOUBLE], local_data, root=0)

    # Initialize centroids on root
    if rank == 0:
        centroids = initialize_centroids(full_data, k)
    else:
        centroids = np.zeros((k, 2))

    comm.Bcast(centroids, root=0)

    for _ in range(max_iters):
        labels = assign_clusters(local_data, centroids)
        local_centroids, local_counts = compute_local_centroids(local_data, labels, k)

        # Gather centroids and counts from all processes
        gathered_centroids = np.zeros((size, k, 2))
        gathered_counts = np.zeros((size, k))

        comm.Gather(local_centroids, gathered_centroids, root=0)
        comm.Gather(local_counts, gathered_counts, root=0)

        # Recalculate global centroids on root
        if rank == 0:
            total_counts = np.sum(gathered_counts, axis=0)
            new_centroids = np.sum(gathered_centroids * gathered_counts[:, :, None], axis=0) / total_counts[:, None]
            if np.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids

        comm.Bcast(centroids, root=0)

    # Final cluster assignment (for visualization)
    final_labels = assign_clusters(local_data, centroids)

    # Gather final results
    all_labels = None
    all_data = None
    if rank == 0:
        all_labels = np.empty(full_data.shape[0], dtype=int)
        all_data = np.empty_like(full_data)

    comm.Gatherv(final_labels,
                 [all_labels, split_sizes, displacements, MPI.INT],
                 root=0)
    comm.Gatherv(local_data,
                 [all_data,
                  tuple(s * 2 for s in split_sizes),
                  tuple(d * 2 for d in displacements),
                  MPI.DOUBLE],
                 root=0)

    if rank == 0:
        plot_clusters(all_data, all_labels, centroids)

if __name__ == "__main__":
    main()
