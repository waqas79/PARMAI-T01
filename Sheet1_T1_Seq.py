import time
import random

#array_x = [2, 4, 6, 8, 1, 3, 5, 7]
array_x = [random.randint(1, 9999) for _ in range(999)]

# Function to calculate the prefix sum in parallel
def Array_prefix_sum(array_x):

    seq_sum = [] # blank list for saving sum
    sum_counter = 0 # init temp_sum
    time_steps = 0
    operations = 0
    for num in array_x:
        sum_counter += num   # adding ( operation +1)
        seq_sum.append(sum_counter) # append ( operation 1+)
        time_steps += 1 # each cycle is a time stemp
        operations += 2 # one cycle 2 operations
    return seq_sum,time_steps,operations
result = Array_prefix_sum(array_x)


start_time = time.time()
result_seq, time_steps, operations = Array_prefix_sum(array_x)
end_time = time.time()




# Display results
print("Input Array_x:", array_x)
print("Prefix sum of array_x:", result)
print("Time steps:", time_steps)
print("Operations:", operations)
print("Number of CPUs:", "1 - Seq Programm")
seq_time = (end_time - start_time) * 1000
print("Sequential:", result_seq, f"Time: {seq_time:.2f} ms")