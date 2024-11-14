import time
from multiprocessing import Pool

array_x = [2, 4, 6, 8, 1, 3, 5, 7]
print("Array_x:", array_x)

# Function to calculate the prefix sum in parallel
def partial_sum(args):
    arr, start, step = args
    """ Helper function to compute partial sums in parallel """
    for i in range(start, len(arr), step):
        if i + step // 2 < len(arr):
            arr[i] += arr[i + step // 2]
    return arr

def parallel_prefix_sum(array_x):
    array_sum = array_x[:]  # Copy the input array to work on
    time_steps = 0           # Counter for time steps
    operations = 0           # Counter for operations
    processors = []          # Track used CPUs (equal to number of parallel jobs)

    n = len(array_sum)
    step = 1  # Initial step size for parallel processing

    # Parallel reduction phase
    while step < n:
        time_steps += 1
        operations += n // step  # Every element addition is counted as an operation
        
        # Properly format the arguments for partial_sum as tuples
        tasks = [(array_sum, i, step * 2) for i in range(0, step)]
        
        # Using pool.map to distribute tasks
        with Pool() as pool:
            processors.append(pool._processes)
            pool.map(partial_sum, tasks)
        
        step *= 2  # Increase step size for next iteration

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
print("Input Array_x:", array_x)
print("Prefix sum of array_x:", result)
print("Time steps:", time_steps)
print("Operations:", operations)
print("Number of required CPUs:", required_cpus)
print("Processing time (ms):", parallel_deltaTime)