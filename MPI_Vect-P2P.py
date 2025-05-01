from mpi4py import MPI
import numpy as np
from datetime import datetime


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
size = comm.Get_size()

if my_rank == 0:
    a = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    b = np.array([6, 7, 8, 9, 10, 11], dtype=np.float64)
    a_list = np.array_split(a, size)
    b_list = np.array_split(b, size)
    
else:
    a_list = None
    b_list = None

# Scatter manually using send/recv
if my_rank == 0:
    # Rank 0 sends slices to other ranks
    for dest in range(1, size):
        comm.send(a_list[dest], dest=dest, tag=0)
        comm.send(b_list[dest], dest=dest, tag=1)
    local_a = a_list[0]
    local_b = b_list[0]
else:
    # Other ranks receive their slice
    local_a = comm.recv(source=0, tag=0)
    local_b = comm.recv(source=0, tag=1)

print(f"Rank {my_rank}: local_a = {local_a}, local_b = {local_b}")

# we loop 100000 times to perform multiplication to simulate a long computation 
# and to make the difference between P2P and collective communication more visible
start_time = datetime.now()
for i in range(100000):
    local_product = local_a * local_b
    local_sum = np.sum(local_product)
    # print(f"Rank {my_rank}: local_sum = {local_sum}")
# P2P gather: send local_sum manually
if my_rank == 0:
    global_sum = local_sum
    for source in range(1, size):
        received_sum = comm.recv(source=source, tag=2)
        global_sum += received_sum
    print(f"Global sum of element-wise product: {global_sum}")
    print(f"Timer {datetime.now() - start_time}")
else:
    comm.send(local_sum, dest=0, tag=2)
