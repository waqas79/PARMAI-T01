import time
import numpy as np
import matplotlib.pyplot as plt
import random
input_Array = [2, 4, 6, 8, 1, 3, 5, 7]
array_lenth = len(input_Array)



from joblib import Parallel, delayed




###################################################################################
#                       T1
##################################################################################
### define ###


def seq_sum(input_Array) :  ### seq_sum    
    seq_ArrayOut = [0] * array_lenth # same lenth 
    seq_ArrayOut[0] = input_Array[0] # ist value just copy
    for i in range (1, array_lenth):
        seq_ArrayOut[i] = seq_ArrayOut[i-1] + input_Array[i] # add last value and last value into instent
    return seq_ArrayOut
def para_sum(input_Array): ### parallel_sum
    para_ArrayOut = np.zeros(array_lenth) # same lenth
    para_ArrayOut[0] = input_Array[0] # ist value just copy 
    for i in range ( 1, array_lenth):
        para_ArrayOut[i] = para_ArrayOut[i-1]+input_Array[i]
    return para_ArrayOut.tolist()
def Vac_sum(input_Array):
    return np.cumsum(input_Array)
####Calc###
para_start_T =  time.time()
print("Para Sum:", Vac_sum(input_Array))
para_end_T = time.time()
para_T = para_end_T - para_start_T

seq_start_T = time.time()
print("Seq Sum:", seq_sum(input_Array))
seq_End_T = time.time()
seq_T =  seq_End_T - seq_start_T 

print("seq Time: ", seq_T)
print("Para Time: ", para_T)

###T1 - Graphics ### 
execution_times = [seq_T, para_T]
labels = ['Sequential', 'Parallel']

# Plot the execution times
plt.figure(figsize=(10, 6))
plt.bar(labels, execution_times, color=['blue', 'green'])
#plt.xlabel('Algorithm')
plt.ylabel('T(s)')
plt.title('T1')
plt.show()

#########################################################################################
###             T2
###############################################################
def scalar_product(A, B):
    return sum(a*b for a, b in zip(A, B))

# Example Vectors
A = np.random.rand(160)
B = np.random.rand(160)

# Measure time for sequential algorithm
start_time = time.time()
sequential_scalar_product = scalar_product(A, B)
sequential_scalar_time = time.time() - start_time

print("Sequential Scalar Product:", sequential_scalar_product)
print("Sequential Scalar Execution Time:", sequential_scalar_time)


#### Parallel Algorithm for Scalar Product using 8 Processors



def parallel_scalar_product(A, B, n_processors=8):
    def sub_scalar_product(i):
        return A[i] * B[i]

    partial_results = Parallel(n_jobs=n_processors)(delayed(sub_scalar_product)(i) for i in range(len(A)))
    return sum(partial_results)

# Measure time for parallel algorithm
start_time = time.time()
parallel_scalar_product_result = parallel_scalar_product(A, B)
parallel_scalar_time = time.time() - start_time

print("Parallel Scalar Product:", parallel_scalar_product_result)
print("Parallel Scalar Execution Time:", parallel_scalar_time)

# Define execution times
scalar_execution_times = [sequential_scalar_time, parallel_scalar_time]
scalar_labels = ['Sequential', 'Parallel']

# Plot the execution times
plt.figure(figsize=(10, 6))
plt.bar(scalar_labels, scalar_execution_times, color=['blue', 'green'])
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (seconds)')
plt.title('Sequential vs Parallel Scalar Product Execution Time')
plt.show()

#############################################################################
###     T#03
##########################################################
def erew_matrix_multiplication(A, B):
    N = len(A)
    C = np.zeros((N, N))

    def compute_element(i, j):
        C[i][j] = sum(A[i][k] * B[k][j] for k in range(N))

    # Parallel computation of each element in C
    for i in range(N):
        for j in range(N):
            compute_element(i, j)

    return C

# Example matrices
A = np.random.rand(4, 4)
B = np.random.rand(4, 4)
C = erew_matrix_multiplication(A, B)
print("Matrix C after EREW PRAM multiplication:\n", C)
def crew_matrix_multiplication(A, B):
    N = len(A)
    C = np.zeros((N, N))

    def compute_element(i, j):
        C[i][j] = sum(A[i][k] * B[k][j] for k in range(N))

    # Parallel computation of each element in C
    for i in range(N):
        for j in range(N):
            compute_element(i, j)

    return C
A = np.random.rand(4, 4)
B = np.random.rand(4, 4)
C = crew_matrix_multiplication(A, B)
print("Matrix C after CREW PRAM multiplication:\n", C)

def crcw_matrix_multiplication(A, B):
    N = len(A)
    C = np.zeros((N, N))

    def compute_element(i, j):
        C[i][j] = sum(A[i][k] * B[k][j] for k in range(N))

    # Parallel computation of each element in C
    for i in range(N):
        for j in range(N):
            compute_element(i, j)

    return C
# Example matrices
A = np.random.rand(4, 4)
B = np.random.rand(4, 4)
C = crcw_matrix_multiplication(A, B)
print("Matrix C after CRCW PRAM multiplication:\n", C)
