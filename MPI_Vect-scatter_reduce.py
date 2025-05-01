from mpi4py import MPI
import numpy as np
from datetime import datetime

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
size = comm.Get_size()

# Only rank 0 initializes the full arrays
if my_rank == 0:
    a = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    b = np.array([6, 7, 8, 9, 10, 11], dtype=np.float64)
    
    a_list = np.array_split(a, size)
    b_list = np.array_split(b, size)
else:
    a_list = None
    b_list = None

# Scatter chunks
local_a = comm.scatter(a_list, root=0)
local_b = comm.scatter(b_list, root=0)

print(f"Rank {my_rank}: local_a = {local_a}, local_b = {local_b}")
start_time = datetime.now()
# Repeat many times to measure time
for i in range(100000):
    local_product = local_a * local_b
    local_sum = np.sum(local_product)

# Reduce local sums into a global sum (root=0)
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# Only root prints the result
if my_rank == 0:
    print(f"Global sum of element-wise product: {global_sum}")
    print(f"Timer {datetime.now() - start_time}")
