import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Sequential Dot Product Algorithm
def sequential_dot_product(A, B):
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

# Compute chunk function
def compute_chunk(start, end, A, B):
    partial_sum = 0
    for i in range(start, end):
        partial_sum += A[i] * B[i]
    return partial_sum

# Parallel Dot Product Algorithm with ProcessPoolExecutor
def parallel_dot_product(A, B, num_processors=4):
    n = len(A)
    chunk_size = n // num_processors
    results = []

    # Step 1: Split work across processors using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        futures = [executor.submit(compute_chunk, i * chunk_size, (i + 1) * chunk_size, A, B) 
                   for i in range(num_processors)]
        results = [future.result() for future in futures]

    # Step 2: Combine results
    total = sum(results)
    return total

# Parallel Dot Product Algorithm with ThreadPoolExecutor (for smaller tasks)
def parallel_dot_product_threads(A, B, num_processors=4):
    n = len(A)
    chunk_size = n // num_processors
    results = []

    # Step 1: Split work across processors using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_processors) as executor:
        futures = [executor.submit(compute_chunk, i * chunk_size, (i + 1) * chunk_size, A, B) 
                   for i in range(num_processors)]
        results = [future.result() for future in futures]

    # Step 2: Combine results
    total = sum(results)
    return total

# Example with a larger vector size
n = 16000000  # Increased size for better performance in parallel
A = np.random.rand(n)
B = np.random.rand(n)

# Measure sequential dot product time
start_time = time.time()
result_sequential = sequential_dot_product(A, B)
T_sequential = time.time() - start_time
print(f"Sequential Dot Product: {result_sequential}")
print(f"Time taken for Sequential Dot Product: {T_sequential:.6f} seconds")

# Measure parallel dot product time (using ProcessPoolExecutor)
start_time = time.time()
result_parallel = parallel_dot_product(A, B, num_processors=4)  # Use 4 processors instead of 8
T_parallel = time.time() - start_time
print(f"Parallel Dot Product (ProcessPoolExecutor): {result_parallel}")
print(f"Time taken for Parallel Dot Product (4 processors): {T_parallel:.6f} seconds")

# Measure parallel dot product time (using ThreadPoolExecutor)
start_time = time.time()
result_parallel_threads = parallel_dot_product_threads(A, B, num_processors=4)  # Use 4 threads
T_parallel_threads = time.time() - start_time
print(f"Parallel Dot Product (ThreadPoolExecutor): {result_parallel_threads}")
print(f"Time taken for Parallel Dot Product (4 threads): {T_parallel_threads:.6f} seconds")

# Calculate Speed-Up (S4) and Efficiency (E4) for ProcessPoolExecutor
S_4 = T_sequential / T_parallel
E_4 = S_4 / 4

# Calculate Speed-Up (S4) and Efficiency (E4) for ThreadPoolExecutor
S_4_threads = T_sequential / T_parallel_threads
E_4_threads = S_4_threads / 4

print(f"\nSpeed-Up (S4) using ProcessPoolExecutor: {S_4:.2f}")
print(f"Efficiency (E4) using ProcessPoolExecutor: {E_4:.2f}")

print(f"\nSpeed-Up (S4) using ThreadPoolExecutor: {S_4_threads:.2f}")
print(f"Efficiency (E4) using ThreadPoolExecutor: {E_4_threads:.2f}")
