from joblib import Parallel, delayed
import numpy as np
import os
import multiprocessing
import time

# Get system information
def get_system_info():
    num_cores = os.cpu_count()  # Total number of cores
    num_physical_cores = multiprocessing.cpu_count()  # Total physical processors
    print(f"Total Number of Cores: {num_cores}")
    print(f"Total Number of Physical Processors: {num_physical_cores}")

# Sequential Matrix Multiplication
def sequential_matrix_multiplication(A, B):
    N = len(A)
    C = [[0] * N for _ in range(N)]  # Resultant matrix initialization
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Parallel Matrix Multiplication Using Joblib
def compute_row(i, A, B):
    return [sum(A[i][k] * B[k][j] for k in range(len(A))) for j in range(len(A))]

def parallel_matrix_multiplication_joblib(A, B):
    N = len(A)
    # Use all cores
    C = Parallel(n_jobs=-1)(delayed(compute_row)(i, A, B) for i in range(N))
    return C

# Test both implementations
N = 200
A = np.random.randint(1, 10, (N, N))
B = np.random.randint(1, 10, (N, N))

# System information
get_system_info()

# Sequential Execution
start_time_seq = time.time()
C_seq = sequential_matrix_multiplication(A.tolist(), B.tolist())
end_time_seq = time.time()
time_seq = end_time_seq - start_time_seq

# Parallel Execution
start_time_par = time.time()
C_par = parallel_matrix_multiplication_joblib(A.tolist(), B.tolist())
end_time_par = time.time()
time_par = end_time_par - start_time_par

# Print Results
print(f"\nSequential Matrix Multiplication Time: {time_seq:.6f} seconds")
print(f"Parallel Matrix Multiplication Time (Joblib): {time_par:.6f} seconds")

# Compare Performance
speedup = time_seq / time_par if time_par > 0 else float('inf')
print(f"Speedup (S_p): {speedup:.2f}")

efficiency = speedup / (os.cpu_count() if os.cpu_count() > 0 else 1)
print(f"Efficiency (E_p): {efficiency:.2f}")
