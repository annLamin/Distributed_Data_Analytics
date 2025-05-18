from mpi4py import MPI
import numpy as np
import logging

logging.basicConfig(format='%(name)s: %(message)s', level=logging.INFO) # level=logging.INFO is essential to use logger.INFO()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger(str(rank))
# Parameters (same across all processes)
dataset_size = 10000
num_features = 10
k = 7
num_classes = 3

# Root process generates the full dataset
if rank == 0:
    synthetic_dataset = np.random.rand(dataset_size, num_features)
    synthetic_labels = np.random.randint(0, num_classes, size=(dataset_size, 1))
    synthetic_dataset = np.append(synthetic_dataset, synthetic_labels, axis=1)
    query_datapoint = np.random.rand(1, num_features)
else:
    synthetic_dataset = None
    query_datapoint = None

# Broadcast the query point to all processes
query_datapoint = comm.bcast(query_datapoint, root=0)

# Divide data among processes
local_data_size = dataset_size // size
remainder = dataset_size % size

# Scatter data manually to account for remainder
counts = [local_data_size + 1 if i < remainder else local_data_size for i in range(size)]
displacements = [sum(counts[:i]) for i in range(size)]

# Prepare buffers
local_count = counts[rank]
local_data = np.empty((local_count, num_features + 1), dtype='float64')

# Scatter dataset
if rank == 0:
    flat_data = synthetic_dataset
else:
    flat_data = None

comm.Scatterv([flat_data,
               tuple(c*(num_features+1) for c in counts),
               tuple(d*(num_features+1) for d in displacements),
               MPI.DOUBLE],
              local_data, root=0)

# Compute local distances
local_distances = np.zeros((local_count, 2))  # [distance, label]
for i in range(local_count):
    dist = np.linalg.norm(query_datapoint - local_data[i, :-1])
    local_distances[i, 0] = dist
    local_distances[i, 1] = local_data[i, -1]
logger.info(f"{local_distances.shape[0]} distances were calculated")

# Gather all distances to root
if rank == 0:
    all_counts = sum(counts)
    all_distances = np.empty((all_counts, 2), dtype='float64')
else:
    all_distances = None

comm.Gatherv(local_distances,
             [all_distances,
              tuple(c*2 for c in counts), # defines number of elements expected from each process
              tuple(d*2 for d in displacements), # define the index elements from each process are stored at
              MPI.DOUBLE], # defines expected datatype
             root=0)

# Root process computes final classification
if rank == 0:
    sorted_indices = np.argsort(all_distances[:, 0])
    top_k_labels = all_distances[sorted_indices[:k], 1].astype(int)
    most_common_label = np.bincount(top_k_labels).argmax()
    logger.info(f"The class of the query instance is {most_common_label}")
