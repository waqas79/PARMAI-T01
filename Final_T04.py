import time
import matplotlib.pyplot as plt
import math
import random

# EREW-PRAM Algorithm
def erew_max(array, num_processors):
    start_time = time.perf_counter()

    # Step 1: Divide array into chunks
    chunk_size = len(array) // num_processors
    local_max = []
    steps = 0  # Count computational steps

    # Step 2: Compute local maximums
    for i in range(num_processors):
        chunk = array[i * chunk_size : (i + 1) * chunk_size]
        local_max.append(max(chunk))
        steps += len(chunk) - 1  # Each chunk requires (chunk_size - 1) comparisons

    # Step 3: Reduce local maximums
    while len(local_max) > 1:
        next_level = []
        for i in range(0, len(local_max), 2):
            if i + 1 < len(local_max):
                next_level.append(max(local_max[i], local_max[i + 1]))
                steps += 1  # One comparison
            else:
                next_level.append(local_max[i])  # Handle odd case
        local_max = next_level

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return local_max[0], execution_time, steps

# CREW-PRAM Algorithm
def crew_max(array, num_processors):
    # Same logic as EREW, concurrent reads only affect implementation on hardware
    return erew_max(array, num_processors)

# CRCW-PRAM Algorithm
def crcw_max(array):
    start_time = time.perf_counter()
    steps = 0  # Count computational steps

    # Step 1: Pairwise comparisons
    while len(array) > 1:
        next_level = []
        for i in range(0, len(array), 2):
            if i + 1 < len(array):
                next_level.append(max(array[i], array[i + 1]))
                steps += 1  # One comparison
            else:
                next_level.append(array[i])  # Handle odd case
        array = next_level

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return array[0], execution_time, steps

# Function to calculate theoretical time complexity
def calculate_time_complexity(n, p):
    # EREW and CREW
    erew_time = n / p + math.log2(p)
    crew_time = n / p + math.log2(p)
    # CRCW
    crcw_time = math.log2(n)
    return erew_time, crew_time, crcw_time

# Test and Graphical Comparison
def compare_algorithms(array, num_processors):
    # Run all algorithms and record times and steps
    erew_result, erew_time, erew_steps = erew_max(array, num_processors)
    crew_result, crew_time, crew_steps = crew_max(array, num_processors)
    crcw_result, crcw_time, crcw_steps = crcw_max(array)

    # Verify correctness
    assert erew_result == crew_result == crcw_result, "Algorithms produced different results!"

    # Calculate theoretical time complexities
    n = len(array)
    erew_theoretical, crew_theoretical, crcw_theoretical = calculate_time_complexity(n, num_processors)

    # Display Results
    print(f"EREW-PRAM Maximum: {erew_result}, Time: {erew_time:.6f} seconds, Steps: {erew_steps}, Theoretical Time Complexity: {erew_theoretical:.2f}")
    print(f"CREW-PRAM Maximum: {crew_result}, Time: {crew_time:.6f} seconds, Steps: {crew_steps}, Theoretical Time Complexity: {crew_theoretical:.2f}")
    print(f"CRCW-PRAM Maximum: {crcw_result}, Time: {crcw_time:.6f} seconds, Steps: {crcw_steps}, Theoretical Time Complexity: {crcw_theoretical:.2f}")

    # Plot Results
    algorithms = ["EREW", "CREW", "CRCW"]
    times = [erew_time, crew_time, crcw_time]
    steps = [erew_steps, crew_steps, crcw_steps]
    theoretical_times = [erew_theoretical, crew_theoretical, crcw_theoretical]

    # Time Comparison Plot
    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, times, color=['blue', 'green', 'orange'], alpha=0.7, label="Measured Time")
    plt.bar(algorithms, theoretical_times, color=['blue', 'green', 'orange'], alpha=0.3, label="Theoretical Time")
    plt.xlabel("PRAM Models")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Comparison of PRAM Models (Measured vs Theoretical Time)")
    plt.legend()
    plt.show()

    # Steps Comparison Plot
    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, steps, color=['blue', 'green', 'orange'])
    plt.xlabel("PRAM Models")
    plt.ylabel("Computational Steps")
    plt.title("Comparison of PRAM Models (Steps)")
    plt.show()

#A =  [random.randint(1, 999) for _ in range(9999)] #
A= [2, 4, 6, 8, 1, 3, 5, 7]
num_processors = 8
compare_algorithms(A, num_processors)
