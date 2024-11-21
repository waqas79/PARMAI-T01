import time

def sequential_scalar_product(A, B):
    start_time = time.time()
    steps = 0  # Step counter
    scalar_product = 0
    for i in range(len(A)):
        scalar_product += A[i] * B[i]
        steps += 1  # One step for each multiplication and addition
    end_time = time.time()
    execution_time = end_time - start_time
    return scalar_product, execution_time, steps

# Test sequential algorithm
A = [i for i in range(16000)]  # Vector A
B = [i * 2 for i in range(16000)]  # Vector B
scalar_product_seq, time_seq, steps_seq = sequential_scalar_product(A, B)
print("Sequential Scalar Product:", scalar_product_seq)
print("Sequential Execution Time:", time_seq)
print("Sequential Steps:", steps_seq)


from concurrent.futures import ThreadPoolExecutor

def parallel_scalar_product(A, B, num_threads=8):
    n = len(A)
    segment_size = (n + num_threads - 1) // num_threads
    results = [0] * num_threads
    steps = 0  # Step counter for each multiplication

    def compute_segment(start, end):
        local_sum = 0
        local_steps = 0
        for i in range(start, end):
            local_sum += A[i] * B[i]
            local_steps += 1  # Each multiplication and addition is a step
        return local_sum, local_steps

    start_time = time.time()

    # Divide the task among threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * segment_size
            end = min((i + 1) * segment_size, n)
            if start < end:
                futures.append(executor.submit(compute_segment, start, end))

        # Collect results
        total_sum = 0
        for future in futures:
            segment_sum, segment_steps = future.result()
            total_sum += segment_sum
            steps += segment_steps

    end_time = time.time()
    execution_time = end_time - start_time
    return total_sum, execution_time, steps, num_threads

# Test parallel algorithm
scalar_product_par, time_par, steps_par, processors = parallel_scalar_product(A, B)
print("Parallel Scalar Product:", scalar_product_par)
print("Parallel Execution Time:", time_par)
print("Parallel Steps:", steps_par)
print("Number of Processors Used:", processors)


# Calculate Speedup and Efficiency
speedup = time_seq / time_par
efficiency = speedup / processors

print("Speedup (S_p):", speedup)
print("Efficiency (E_p):", efficiency)
import numpy as np
import time

def optimized_parallel_scalar_product(A, B):
    start_time = time.time()
    scalar_product = np.dot(A, B)  # Efficient vectorized operation
    end_time = time.time()
    execution_time = end_time - start_time
    return scalar_product, execution_time

# Test the optimized parallel implementation
A = np.arange(16000)  # Vector A
B = np.arange(16000) * 2  # Vector B
optimized_result, optimized_time = optimized_parallel_scalar_product(A, B)
print("Optimized Parallel Scalar Product:", optimized_result)
print("Optimized Parallel Execution Time:", optimized_time)

def parallel_scalar_product_dynamic(A, B, num_threads=8):
    n = len(A)
    segment_size = (n + num_threads - 1) // num_threads
    results = [0] * num_threads
    steps = 0  # Step counter

    def compute_segment(start, end):
        local_sum = 0
        local_steps = 0
        for i in range(start, end):
            local_sum += A[i] * B[i]
            local_steps += 1
        return local_sum, local_steps

    start_time = time.time()

    # Dynamically allocate tasks
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start = i * segment_size
            end = min((i + 1) * segment_size, n)
            if start < end:
                futures.append(executor.submit(compute_segment, start, end))

        total_sum = 0
        for future in futures:
            segment_sum, segment_steps = future.result()
            total_sum += segment_sum
            steps += segment_steps

    end_time = time.time()
    execution_time = end_time - start_time
    return total_sum, execution_time, steps, num_threads

# Test the dynamically balanced parallel implementation
dynamic_result, dynamic_time, dynamic_steps, dynamic_processors = parallel_scalar_product_dynamic(A, B)
print("Dynamically Balanced Parallel Scalar Product:", dynamic_result)
print("Dynamic Parallel Execution Time:", dynamic_time)
print("Dynamic Parallel Steps:", dynamic_steps)
def measure_scalability(A, B):
    max_threads = 8
    sequential_result, sequential_time, _ = sequential_scalar_product(A, B)
    speedups = []
    efficiencies = []

    for threads in range(1, max_threads + 1):
        parallel_result, parallel_time, _, _ = parallel_scalar_product_dynamic(A, B, num_threads=threads)
        speedup = sequential_time / parallel_time
        efficiency = speedup / threads
        speedups.append(speedup)
        efficiencies.append(efficiency)
        print(f"Threads: {threads}, Speedup: {speedup:.2f}, Efficiency: {efficiency:.2f}")

    return speedups, efficiencies

# Measure scalability
speedups, efficiencies = measure_scalability(A, B)
import matplotlib.pyplot as plt

# Plot speedup and efficiency
def visualize_scalability(speedups, efficiencies):
    threads = range(1, len(speedups) + 1)

    plt.figure(figsize=(10, 5))

    # Speedup
    plt.subplot(1, 2, 1)
    plt.plot(threads, speedups, marker="o")
    plt.title("Speedup vs. Number of Threads")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.grid(True)

    # Efficiency
    plt.subplot(1, 2, 2)
    plt.plot(threads, efficiencies, marker="o")
    plt.title("Efficiency vs. Number of Threads")
    plt.xlabel("Number of Threads")
    plt.ylabel("Efficiency")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Visualize results
visualize_scalability(speedups, efficiencies)
