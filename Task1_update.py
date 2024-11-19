import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Sequential Prefix Sum Algorithm
def sequential_prefix_sum(x):
    n = len(x)
    # Initialize the prefix sum array
    S = np.zeros(n, dtype=int)
    
    # First element is just the same
    S[0] = x[0]
    
    # Compute the prefix sum for the rest of the elements
    for i in range(1, n):
        S[i] = S[i - 1] + x[i]
    
    return S

# Parallel Prefix Sum Algorithm using up-sweep and down-sweep
def up_sweep(x, start, step):
    """Perform the up-sweep (reduction) phase in parallel."""
    for i in range(start, len(x), step):
        x[i] += x[i - step]
    return x

def down_sweep(x, start, step):
    """Perform the down-sweep phase (propagation) in parallel."""
    for i in range(start, len(x), step):
        temp = x[i]
        x[i] = x[i - step]
        x[i - step] += temp
    return x

def parallel_prefix_sum(arr):
    n = len(arr)
    num_threads = 4
    chunk_size = n // num_threads
    chunks = []

    def compute_chunk(start, end):
        chunk = arr[start:end]
        for i in range(1, len(chunk)):
            chunk[i] += chunk[i - 1]
        return chunk

    # Step 1: Compute prefix sum for each chunk
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_chunk, i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]
        chunks = [future.result() for future in futures]

    # Step 2: Adjust for carry-over between chunks
    for i in range(1, len(chunks)):
        # Add the last element of the previous chunk to all elements in the current chunk
        carry = chunks[i-1][-1]
        for j in range(len(chunks[i])):
            chunks[i][j] += carry

    # Step 3: Merge the chunks into a final result
    result = np.concatenate(chunks)
    return result
# Test data
x = np.array([2, 4, 6, 8, 1, 3, 5, 7])

# Get results from both algorithms

result_sequential = sequential_prefix_sum(x)
result_parallel = parallel_prefix_sum(x)

# Print results (optional)
print("__input___:", x)
print("Sequential Prefix Sum:", result_sequential)
print("Parallel Prefix Sum:", result_parallel)

# Plotting
plt.figure(figsize=(10, 6))

# Plot Sequential Prefix Sum
plt.subplot(2, 1, 1)  # Two rows, one column, first subplot
plt.plot(range(len(x)), result_sequential, marker='o', color='b', label='Sequential', linestyle='-', markersize=8)
plt.title('Sequential Prefix Sum')
plt.xlabel('Index')
plt.ylabel('Prefix Sum')
plt.xticks(range(len(x)))
plt.grid(True)
plt.legend()

# Plot Parallel Prefix Sum
plt.subplot(2, 1, 2)  # Two rows, one column, second subplot
plt.plot(range(len(x)), result_parallel, marker='x', color='r', label='Parallel', linestyle='-', markersize=8)
plt.title('Parallel Prefix Sum')
plt.xlabel('Index')
plt.ylabel('Prefix Sum')
plt.xticks(range(len(x)))
plt.grid(True)
plt.legend()

# Adjust layout
plt.tight_layout()
plt.show()