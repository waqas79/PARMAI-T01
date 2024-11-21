import time
from concurrent.futures import ThreadPoolExecutor

# Sequential Prefix Sum with time and step calculation
def sequential_prefix_sum(array):
    start_time = time.time()
    steps = 0  # Step counter
    prefix_sum = [0] * len(array)
    prefix_sum[0] = array[0]
    steps += 1  # Initialization step
    for i in range(1, len(array)):
        prefix_sum[i] = prefix_sum[i - 1] + array[i]
        steps += 1  # Step for each addition
    end_time = time.time()
    execution_time = end_time - start_time
    return prefix_sum, execution_time, steps

# Parallel Prefix Sum with time and step calculation
def parallel_prefix_sum(array, num_threads=4):
    n = len(array)
    segment_size = (n + num_threads - 1) // num_threads
    results = [0] * n
    steps = 0  # Step counter for additions

    def compute_segment(start, end):
        segment_sum = [0] * (end - start)
        segment_sum[0] = array[start]
        local_steps = 1  # Initialization step
        for i in range(1, end - start):
            segment_sum[i] = segment_sum[i - 1] + array[start + i]
            local_steps += 1  # Step for each addition
        return segment_sum, local_steps

    start_time = time.time()

    # Step 1: Compute local prefix sums
    segments = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * segment_size
            end = min((i + 1) * segment_size, n)
            if start < end:
                segments.append((start, end))
        futures = {executor.submit(compute_segment, start, end): (start, end) for start, end in segments}

    partial_sums = {}
    for future in futures:
        start, end = futures[future]
        segment, local_steps = future.result()
        partial_sums[start] = (segment, local_steps)
        steps += local_steps

    # Step 2: Adjust segments
    current_offset = 0
    for start, (segment, _) in sorted(partial_sums.items()):
        for i in range(len(segment)):
            results[start + i] = segment[i] + current_offset
            steps += 1  # Adjusting step
        current_offset += segment[-1]

    end_time = time.time()
    execution_time = end_time - start_time
    return results, execution_time, steps, num_threads

# Testing both algorithms
x = [2, 4, 6, 8, 1, 3, 5, 7]

# Sequential computation
seq_result, seq_time, seq_steps = sequential_prefix_sum(x)
print("Sequential Prefix Sum:", seq_result)
print("Sequential Execution Time:", seq_time)
print("Sequential Steps:", seq_steps)

# Parallel computation
par_result, par_time, par_steps, processors = parallel_prefix_sum(x, num_threads=8)
print("Parallel Prefix Sum:", par_result)
print("Parallel Execution Time:", par_time)
print("Parallel Steps:", par_steps)
print("Number of Processors Used:", processors)
