# First example
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
size = comm.Get_size()
worker = comm.Get_rank()
# print(f"MPI: rank {comm.Get_rank()} | {comm.Get_size()}")
print(f"MPI: rank {worker} | size : {size}")


# second example
import logging
logging.basicConfig(format ='%(name)s: %(message)s')
my_rank = comm.Get_rank()
logger = logging.getLogger(f"rank {my_rank}")
logger.warning(f"hello")
logger.warning(f"hello : rank {logger}")

# third example (Deadlocks)
# if my_rank == 0:
#     msg = comm.recv(source=1)
#     logger.warning(f"received message: {msg}")
# else: # the chunck of code that rank 1 executes
#     msg = comm.recv(source=0)
#     msg = f"hello from rank {my_rank}"
#     comm.send(msg, dest=0)
#     logger.warning(f"sent message: {msg}")


# fourth example
if my_rank == 0:
    data = np.arange(9, dtype='int32')
    # print('data :' ,data)
    data_list = np.array_split(data, comm.Get_size()) # returns a list of numpy ndarrays
else:
    data_list = None
my_data = comm.scatter(data_list, root=0)
logger.warning(f"my_data: {my_data} of type {type(my_data)}")

# fifth example
my_data = my_data * my_rank
print(my_data)
results = comm.gather(my_data, root=0)
if my_rank == 0:
    logger.warning(f"results: {results}")
else:
    logger.warning(f"results: {results}")

# Vector multiplication

if my_rank == 0:
    list1 = [1,2,3,4]
    list2 = [3,4,5,6]
else:
    list1 = None
    list2 = None
    
my_list1 = comm.scatter(list1, root=0)
my_list2 = comm.scatter(list2, root=0)
logger.warning(f"my_data: {my_list1} of type {type(my_list1)}")

matr_mult =  my_list1 * my_list2
    








