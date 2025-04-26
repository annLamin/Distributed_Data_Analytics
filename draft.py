# First example
from mpi4py import MPI
comm = MPI.COMM_WORLD
# print(f"MPI: rank {comm.Get_rank()} | {comm.Get_size()}")


# # second example
import logging
logging.basicConfig(format ='%(name)s: %(message)s')
my_rank = comm.Get_rank()
logger = logging.getLogger(f"rank {my_rank}")
logger.warning(f"hello")
# logger.warning(f"hello : rank {logger}")

# third example
if my_rank == 0:
    msg = comm.recv(source=1)
    logger.warning(f"received message: {msg}")
else: # the chunck of code that rank 1 executes
    msg = comm.recv(source=0)
    msg = f"hello from rank {my_rank}"
    comm.send(msg, dest=0)
    logger.warning(f"sent message: {msg}")
