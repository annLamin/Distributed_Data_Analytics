from mpi4py import MPI
import numpy as np
import logging
from datetime import datetime

# Setup basic logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
size = comm.Get_size()

# Only rank 0 initializes the full arrays
if my_rank == 0:
    a = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    b = np.array([6, 7, 8, 9, 10, 11], dtype=np.float64)
    
    a_list = np.array_split(a, size)  # split into chunks
    b_list = np.array_split(b, size)
else:
    a_list = None
    b_list = None

# Scatter chunks
local_a = comm.scatter(a_list, root=0)
local_b = comm.scatter(b_list, root=0)

logger.warning(f"Rank {my_rank}: local_a = {local_a}, local_b = {local_b}")

# Start timing
start_time = datetime.now()

# Repeat many times to make timing meaningful
for _ in range(100000):
    # Local computation: multiply and sum
    local_product = local_a * local_b
    local_sum = np.sum(local_product)

# Gather local sums at root
local_sums = comm.gather(local_sum, root=0)

# Root process computes final global sum
if my_rank == 0:
    global_sum = sum(local_sums)
    
    logger.warning(f"Full Vector a: {a}")
    logger.warning(f"Full Vector b: {b}")
    logger.warning(f"Local sums from each process: {local_sums}")
    logger.warning(f"Global sum of element-wise product: {global_sum}")
    print(f"Took {datetime.now() - start_time}")
