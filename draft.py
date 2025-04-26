from mpi4py import MPI
comm = MPI.COMM_WORLD
# print(f"MPI: rank {comm.Get_rank()} | {comm.Get_size()}")