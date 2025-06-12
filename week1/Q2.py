import numpy as np

# Create an array of 20 random floats between 0 and 10
random_array = np.random.uniform(0, 10, 20)

# Print the original array rounded to 2 decimal places
print("Original array:")
print(np.round(random_array, 2))

# Calculate statistics
min_val = np.min(random_array)
max_val = np.max(random_array)
median_val = np.median(random_array)

# Print the results
print("\nMinimum value:", round(min_val, 2))
print("Maximum value:", round(max_val, 2))
print("Median value:", round(median_val, 2))

# Create a copy to avoid modifying the original array
modified_array = random_array.copy()

# Replace elements < 5 with their squares
modified_array[modified_array < 5] = modified_array[modified_array < 5] ** 2

# Print the modified array
print("\nArray after replacing elements < 5 with their squares:")
print(np.round(modified_array, 2))

def numpy_alternate_sort(array):
    # Sort the array in ascending order
    sorted_asc = np.sort(array)
    
    # Create an empty array of the same size
    result = np.empty_like(array)
    
    # Initialize pointers for smallest and largest elements
    left = 0
    right = len(array) - 1
    
    # Fill the result array in alternating pattern
    for i in range(len(array)):
        if i % 2 == 0:
            # Even index: take from the left (smallest remaining)
            result[i] = sorted_asc[left]
            left += 1
        else:
            # Odd index: take from the right (largest remaining)
            result[i] = sorted_asc[right]
            right -= 1
    
    return result

# Apply the alternate sort
alternate_sorted = numpy_alternate_sort(random_array)

# Print the result
print("\nAlternately sorted array:")
print(np.round(alternate_sorted, 2))
