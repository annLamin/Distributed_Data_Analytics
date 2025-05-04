from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def selection_sort(arr):
    for i in range(len(arr)):
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

if rank == 0:
    my_list = list(range(10000))
    np.random.shuffle(my_list)
    chunks = np.array_split(my_list, size)
else:
    chunks = None

local_chunk = comm.scatter(chunks, root=0)
local_chunk = selection_sort(list(local_chunk))

sorted_chunks = comm.gather(local_chunk, root=0)

if rank == 0:
    pointers = [0] * size
    final_sorted = []

    while True:
        min_val = None
        min_index = -1

        for i in range(size):
            pos = pointers[i]
            if pos < len(sorted_chunks[i]):
                current_value = sorted_chunks[i][pos]
                if min_val is None or current_value < min_val:
                    min_val = current_value
                    min_index = i
            # print(f"Rank {rank} checking chunk {i}: {sorted_chunks[i]} with pointer {pointers[i]}")
            # print(final_sorted)
        if min_index == -1:
            break
        
        final_sorted.append(min_val)
        pointers[min_index] += 1
    print("Final sorted list:", final_sorted)  
