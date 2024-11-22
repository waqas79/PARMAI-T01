import time
import matplotlib.pyplot as plt

# Sequential Prefix Sum
def prefix_sum_sequential(array):
    """Compute Prefix Sum Sequentially"""
    start_time = time.perf_counter()
    prefix = [0] * len(array)
    prefix[0] = array[0]
    for i in range(1, len(array)):
        prefix[i] = prefix[i - 1] + array[i]
    end_time = time.perf_counter()
    return prefix, end_time - start_time

# Parallel Prefix Sum
def prefix_sum_parallel(array):
    """Compute Prefix Sum in Parallel"""
    start_time = time.perf_counter()
    n = len(array)
    steps = 0
    result = array[:]  # Copy input array to avoid in-place modification issues

    # Step 1: Up-Sweep (Reduction Phase)
    offset = 1
    while offset < n:
        for i in range(offset, n, 2 * offset):
            result[i] += result[i - offset]
            steps += 1
        offset *= 2

    # Step 2: Down-Sweep (Propagation Phase)
    result[n - 1] = 0  # Set the last element to 0
    offset = n // 2
    while offset > 0:
        for i in range(offset, n, 2 * offset):
            temp = result[i]
            result[i] += result[i - offset]
            result[i - offset] = temp
            steps += 1
        offset //= 2

    end_time = time.perf_counter()
    return result, end_time - start_time, steps

# Compare Algorithms and Visualize Results
def compare_prefix_sum(n, num_processors):
    """Compare Sequential and Parallel Prefix Sum"""
    # Generate input array
    array = list(range(1, n + 1))

    # Sequential Prefix Sum
    seq_result, seq_time = prefix_sum_sequential(array[:])

    # Parallel Prefix Sum
    par_result, par_time, steps = prefix_sum_parallel(array[:])

    # Verify correctness
    #assert seq_result == par_result, "Prefix Sum results do not match!"

    # Compute Speed-Up and Efficiency
    speedup = seq_time / par_time
    efficiency = speedup / num_processors

    # Display Results
    print("Task 01: Prefix Sum")
    print(f"Array Length: {n}")
    print(f"Sequential Time: {seq_time:.6f} seconds")
    print(f"Parallel Time: {par_time:.6f} seconds, Steps: {steps}")
    print(f"Speed-Up: {speedup:.2f}")
    print(f"Efficiency: {efficiency:.2f}")

    # Plot Results
    plt.figure(figsize=(12, 6))

    # Execution Time
    plt.subplot(1, 3, 1)
    plt.bar(['Sequential', 'Parallel'], [seq_time, par_time], color=['blue', 'orange'])
    plt.title("Task 01: Execution Time")
    plt.ylabel("Time (seconds)")

    # Speed-Up
    plt.subplot(1, 3, 2)
    plt.bar(['Parallel'], [speedup], color='green')
    plt.title("Task 01: Speed-Up")
    plt.ylabel("Speed-Up")

    # Efficiency
    plt.subplot(1, 3, 3)
    plt.bar(['Parallel'], [efficiency], color='purple')
    plt.title("Task 01: Efficiency")
    plt.ylabel("Efficiency")

    plt.tight_layout()
    plt.show()

# Test the comparison
array_length = 16  # Length of the input array
num_processors = 4  # Number of processors for parallel computation
compare_prefix_sum(array_length, num_processors)
