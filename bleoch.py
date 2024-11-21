import time
import numpy as np
import math
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
	
def sum_blelloch(input_Array):
	
    log_n = math.ceil(math.log2(array_lenth))
    ps = np.zeros(array_lenth, dtype=int)
    ps[:array_lenth] = input_Array

    # Up-sweep phase
    step = 1
    while step < array_lenth:
        for i in range(0, array_lenth, step * 2):
            if i + step * 2 - 1 < array_lenth:
                ps[i + step * 2 - 1] += ps[i + step - 1]
        step *= 2

    # Down-sweep phase
    ps[array_lenth - 1] = 0
    step = array_lenth // 2
    while step > 0:
        for i in range(0, array_lenth, step * 2):
            if i + step * 2 - 1 < array_lenth:
                t = ps[i + step - 1]
                ps[i + step - 1] = ps[i + step * 2 - 1]
                ps[i + step * 2 - 1] += t
        step //= 2

    # Correct the final output to compute actual prefix sums
    prefix_sum_result = np.zeros(array_lenth, dtype=int)
    prefix_sum_result[0] = input_Array[0]
    for i in range(1, array_lenth):
        prefix_sum_result[i] = prefix_sum_result[i - 1] + input_Array[i]
    
    return prefix_sum_result.tolist()
####Calc###
para_start_T =  time.time()
print("Para Sum:", sum_blelloch(input_Array))
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