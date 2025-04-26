from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the full vectors
full_x = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
full_y = np.array([6, 7, 8, 9, 10, 11], dtype=np.float64)
n = len(full_x)

# Calculate the size of each chunk
chunk_size = n // size
remainder = n % size

# Determine the start and end indices for the local chunk
start = rank * chunk_size + min(rank, remainder)
end = (rank + 1) * chunk_size + min(rank + 1, remainder)

# Distribute the data using slicing
local_x = full_x[start:end]
local_y = full_y[start:end]

# Perform local multiplication and calculate the local sum
local_result = local_x * local_y
local_sum = np.sum(local_result)
print(f"Rank {rank}: Local sum = {local_sum}")

global_sum = 0.0
if rank == 0:
    global_sum = local_sum
    for i in range(1, size):
        received_sum = comm.recv(source=i, tag=11)
        global_sum += received_sum
    print("Full Vector x:", full_x)
    print("Full Vector y:", full_y)
    print("Global sum of the element-wise product:", global_sum)
elif rank != size - 1:
    comm.send(local_sum, dest=rank + 1, tag=11)
else:  # Last rank
    comm.send(local_sum, dest=0, tag=11)

MPI.Finalize()