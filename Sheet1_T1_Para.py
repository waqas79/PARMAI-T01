import numpy as np
import dask.array as da
import multiprocessing as mp
import time
import random

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

def prefix_sum_chunk(chunk, offset=0):  # Define the function to include offset
    return np.cumsum(chunk) + offset  # Add offset to adjust for the previous chunk's sum

def prefix_sum_multiprocessing(array_x, num_processes):
    chunk_size = len(array_x) // num_processes
    chunks = [array_x[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
    
    # Create a pool to process the chunks
    with mp.Pool(num_processes) as pool:
        # First, calculate prefix sums for each chunk
        results = []
        offsets = [0] * num_processes
        # Set the offset for each chunk based on the sum of previous chunks
        for i in range(1, num_processes):
            offsets[i] = np.sum(chunks[i-1])  # The offset is the sum of the previous chunk
        
        # Apply the prefix_sum_chunk function to each chunk with its corresponding offset
        results = pool.starmap(prefix_sum_chunk, zip(chunks, offsets))
    
    # Combine the results
    result = np.concatenate(results)
    return result

# Example usage
if __name__ == '__main__':
    array_x = [random.randint(1, 99) for _ in range(5)]
    print("Original array:", array_x)

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
    num_processes = 4  # Use number of available CPU cores
    start_time = time.time()
    result_mp = prefix_sum_multiprocessing(array_x, num_processes)
    end_time = time.time()
    mp_time = (end_time - start_time) * 1000

    # Print results
    print("Sequential:", result_seq, f"Time: {seq_time:.2f} ms")
    print("NumPy:", result_numpy, f"Time: {numpy_time:.2f} ms")
    print("Dask:", result_dask, f"Time: {dask_time:.2f} ms")
    print("Multiprocessing:", result_mp, f"Time: {mp_time:.2f} ms")
