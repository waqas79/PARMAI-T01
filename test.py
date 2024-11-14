import numpy as np
import multiprocessing as mp
import time
import random

# Up-sweep (reduction) phase for calculating partial sums
def up_sweep_chunk(chunk):
    # We want to sum up chunks in a tree structure, but each chunk is processed independently first.
    n = len(chunk)
    for d in range(0, int(np.log2(n)) + 1):  # Perform up-sweep for log2(n) steps
        for i in range(2**d, n, 2**(d + 1)):  # Update each node in the tree
            chunk[i] += chunk[i - 2**d]
    return chunk

# Down-sweep phase for calculating the final prefix sum
def down_sweep_chunk(chunk):
    n = len(chunk)
    # Start from the last element and propagate the partial sums backward
    chunk[-1] = 0  # Set the last element to 0 (identity element)
    for d in range(int(np.log2(n)) - 1, -1, -1):  # Perform down-sweep in reverse order
        for i in range(2**d, n, 2**(d + 1)):
            temp = chunk[i - 2**d]
            chunk[i - 2**d] = chunk[i]
            chunk[i] += temp
    return chunk

# Function to compute the prefix sum in parallel
def prefix_sum_parallel(array_x, num_processes):
    # Split the array into chunks
    chunks = np.array_split(array_x, num_processes)
    
    # Up-sweep phase: Apply reduction on each chunk
    with mp.Pool(num_processes) as pool:
        up_sweep_results = pool.map(up_sweep_chunk, chunks)

    # Combine results from up-sweep
    result = np.concatenate(up_sweep_results)

    # Down-sweep phase: Apply down-sweep to get the final prefix sums
    with mp.Pool(num_processes) as pool:
        down_sweep_results = pool.map(down_sweep_chunk, np.array_split(result, num_processes))

    # Final result after down-sweep
    final_result = np.concatenate(down_sweep_results)
    return final_result

# Generate a random array of 9999 elements
array_x = [random.randint(1, 99) for _ in range(9999)]

# Set number of processes (using the number of CPU cores)
num_processes = mp.cpu_count()

start_time = time.time()
result_mp = prefix_sum_parallel(array_x, num_processes)
end_time = time.time()

mp_time = (end_time - start_time) * 1000  # Time in milliseconds

print(f"Prefix Sum Result: {result_mp}")
print(f"Multiprocessing time: {mp_time:.2f} ms")
