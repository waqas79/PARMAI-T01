import time
def prefix_sum_sequential(array_x):
    prefix_sum = []
    sum_counter = 0
    for num in array_x:
        sum_counter += num
        prefix_sum.append(sum_counter)
    return prefix_sum
def parallel_prefix_sum(arr):
    n = len(arr)

    # Base case: If the array has only one element, return the array itself
    if n == 1:
        return arr

    # Divide the array into two halves
    mid = n // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Recursively calculate the prefix sum for each half
    left_prefix_sum = parallel_prefix_sum(left_half)
    right_prefix_sum = parallel_prefix_sum(right_half)

    # Calculate the sum of the left half
    left_sum = left_prefix_sum[-1]

    # Add the sum of the left half to each element of the right half
    right_prefix_sum = [x + left_sum for x in right_prefix_sum]

    # Combine the results from both halves
    return left_prefix_sum + right_prefix_sum

# Example usage
if __name__ == '__main__':
    x = [2, 4, 6, 8, 1, 3, 5, 7]
    start_time = time.time()
    print("Sequential result:", prefix_sum_sequential(x))
    end_time =  time.time()
    seq_time = (end_time - start_time) * 10
    print("seq time : ",seq_time)
    
    start_time = time.time()
    print("Parallel result:", parallel_prefix_sum(x))
    end_time =  time.time()
    Para_time = (end_time - start_time) * 10
    print ("Para-Time", Para_time)