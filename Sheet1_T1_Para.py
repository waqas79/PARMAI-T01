import time
from multiprocessing import Pool

# Input array
array_x = [2, 4, 6, 8, 1, 3, 5, 7]
print("Array_x:", array_x)

# Helper function to perform partial prefix sum for a section of the array
def partial_sum(args):
    arr, start, step = args
    for i in range(start, len(arr), step):
        if i + step // 2 < len(arr):
            arr[i] += arr[i + step // 2]
    return arr

def parallel_prefix_sum(array_x):
    array_sum = array_x[:]
    time_steps = 0        # Counter for time steps
    operations = 0        # Counter for operations
    processors = []       # Track used CPUs (equal to number of parallel jobs)

    n = len(array_sum)
    step = 1

    # Parallel reduction phase
    while step < n:
        time_steps += 1
        operations += n // step  # Every element addition is counted as an operation
        with Pool() as pool:
            # Divide work among processors in parallel
            processors.append(pool._processes)
            pool.map(partial_sum, [(array_sum, i, step * 2) for i in range(0, step)])
        step *= 2

    # Down-sweep phase (can be parallelized but simpler in sequence here)
    for i in range(n - 1, 0, -1):
        array_sum[i] += array_sum[i - 1]
        operations += 1  # Count the down-sweep additions

    return array_sum, time_steps, operations, max(processors)

# Record start time
start_time = time.time()

# Execute the function and capture the results
result, time_steps, operations, required_cpus = parallel_prefix_sum(array_x)

# Record end time
end_time = time.time()

# Calculate the duration in milliseconds
parallel_deltaTime = (end_time - start_time) * 1e3  # Convert time to milliseconds

# Display results
print("Prefix sum of array_x:", result)
print("Number of time steps:", time_steps)
print("Number of operations:", operations)
print("Number of required CPUs:", required_cpus)
print("Execution time (in milliseconds):", parallel_deltaTime)
