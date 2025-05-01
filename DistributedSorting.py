from mpi4py import MPI
import math
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Example data
data = list(range(10000))
np.random.shuffle(data)

min_data = min(data)
max_data = max(data)

# Bin size calculation
bin_size = math.ceil((max_data - min_data) / size)

# Distribute data to processors (only on rank 0)
new_list = None
if rank == 0:
    new_list = [[] for i in range(size)]
    for x in data:
        bin_index = min(size - 1, math.floor((x - min_data) / bin_size))
        new_list[bin_index].append(x)

# Scatter the lists among the processors
local_data = comm.scatter(new_list, root=0)

# each chunk is sorted locally on its own processor
local_data.sort()

# Gather the sorted lists back to the root process
sorted_chunks = comm.gather(local_data, root=0)

if rank == 0:
    # Combine the sorted lists
    sorted_data = []
    if sorted_chunks:
        for chunk in sorted_chunks:
            sorted_data.extend(chunk)
print("Sorted Data", sorted_data) 
