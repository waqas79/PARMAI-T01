array_x = [2, 4, 6, 8, 1, 3, 5, 7]
print("Array_x : ",array_x)

def Array_prefix_sum(array_x):
    sum = [] # blank list for saving sum
    sum_counter = 0 # init temp_sum

    for num in array_x:
        sum_counter += num
        sum.append(sum_counter)
    return sum



result = Array_prefix_sum(array_x)
print("prefix sum of array_x", result)
