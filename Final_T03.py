import time
import random
import matplotlib.pyplot as plt

# Matrix Multiplication Algorithms

def matrix_multiply_sequential(A, B):
    """Sequential Matrix Multiplication"""
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_multiply_crc(A, B):
    """CRCW PRAM Matrix Multiplication"""
    start_time = time.perf_counter()
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    steps = 0  # Count computational steps

    # Parallel computation for each C[i][j]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                steps += 1

    end_time = time.perf_counter()
    return C, end_time - start_time, steps

# Performance Comparison and Visualization

def compare_and_visualize(matrix_size, num_processors):
    """Compare Sequential and CRCW Multiplication and Visualize Results"""
    # Generate random matrices
    A = [[random.randint(1, 10) for _ in range(matrix_size)] for _ in range(matrix_size)]
    B = [[random.randint(1, 10) for _ in range(matrix_size)] for _ in range(matrix_size)]

    # Sequential Algorithm
    start_seq = time.perf_counter()
    C_seq = matrix_multiply_sequential(A, B)
    seq_time = time.perf_counter() - start_seq

    # CRCW PRAM Algorithm
    C_crc, crc_time, steps = matrix_multiply_crc(A, B)

    # Verify correctness
    assert C_seq == C_crc, "Matrix results do not match!"

    # Compute Speed-Up and Efficiency
    speedup = seq_time / crc_time
    efficiency = speedup / num_processors

    # Display Results
    print(f"Matrix Size: {matrix_size}x{matrix_size}")
    print(f"Sequential Time: {seq_time:.6f} seconds")
    print(f"CRCW PRAM Time: {crc_time:.6f} seconds, Steps: {steps}")
    print(f"Speed-Up: {speedup:.2f}")
    print(f"Efficiency: {efficiency:.2f}")

    # Plot Results
    plt.figure(figsize=(12, 6))

    # Execution Time
    plt.subplot(1, 3, 1)
    plt.bar(['Sequential', 'CRCW PRAM'], [seq_time, crc_time], color=['blue', 'orange'])
    plt.title("Task 3 - Execution Time")
    plt.ylabel("Time (seconds)")

    # Speed-Up
    plt.subplot(1, 3, 2)
    plt.bar(['CRCW PRAM'], [speedup], color='green')
    plt.title("Task 3 - Speed-Up")
    plt.ylabel("Speed-Up")

    # Efficiency
    plt.subplot(1, 3, 3)
    plt.bar(['CRCW PRAM'], [efficiency], color='purple')
    plt.title("Task 3 - Efficiency")
    plt.ylabel("Efficiency")

    plt.tight_layout()
    plt.show()

# Run Comparison and Visualization
matrix_size = 4  # Size of the matrix (e.g., 4x4)
num_processors = 8  # Number of processors for CRCW PRAM
compare_and_visualize(matrix_size, num_processors)
