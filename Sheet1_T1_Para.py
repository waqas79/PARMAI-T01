import numpy as np
import dask.array as da
import multiprocessing as mp
import time
import random
from multiprocessing import pool

def prefix_sum_sequential(array_x):
    seq_sum = []
    sum_counter = 0
    for num in array_x:
        sum_counter += num
        seq_sum.append(sum_counter)
    return seq_sum

def prefix_sum_numpy(array_x):
    return np.cumsum(array_x)

def prefix_sum_dask(array_x):
    dask_array = da.from_array(array_x)
    return da.cumsum(dask_array).compute()

def prefix_sum_chunk(chunk):  # Define the missing function here
    return np.cumsum(chunk)  # Calculate prefix sum for each chunk
def prefix_sum_multiprocessing(array_x, num_processes):
    chunk_size = len(array_x) // num_processes
    chunks = [array_x[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]

    

    with mp.Pool(num_processes) as pool:
        results = pool.map(prefix_sum_chunk, chunks)

    # Combine the results
    result = np.concatenate(results)
    return result

# Example usage
#array_x = [2, 4, 6, 8, 1, 3, 5, 7]
array_x = [random.randint(1, 9999) for _ in range(999)]
print(array_x)

# Sequential
start_time = time.time()
result_seq = prefix_sum_sequential(array_x)
end_time = time.time()
seq_time = (end_time - start_time) * 1000  # milliseconds

# NumPy
start_time = time.time()
result_numpy = prefix_sum_numpy(array_x)
end_time = time.time()
numpy_time = (end_time - start_time) * 1000

# Dask
start_time = time.time()
result_dask = prefix_sum_dask(array_x)
end_time = time.time()
dask_time = (end_time - start_time) * 1000

# Multiprocessing

num_processes = mp.cpu_count()  # Using number of available CPU cores
start_time = time.time()
result_mp = prefix_sum_multiprocessing(array_x, num_processes)
end_time = time.time()
mp_time = (end_time - start_time) * 1000

# Print results
print("Sequential:", result_seq, f"Time: {seq_time:.2f} ms")
print("NumPy:", result_numpy, f"Time: {numpy_time:.2f} ms")
print("Dask:", result_dask, f"Time: {dask_time:.2f} ms")
print("Multiprocessing:", result_mp, f"Time: {mp_time:.2f} ms")
print("Sequential:", f"Time: {seq_time:.2f} ms")
print("NumPy:",  f"Time: {numpy_time:.2f} ms")
print("Dask:",  f"Time: {dask_time:.2f} ms")
print("Multiprocessing:", f"Time: {mp_time:.2f} ms")