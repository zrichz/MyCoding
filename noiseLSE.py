import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# Step 1: Generate 10 tiny (10x10) noise arrays
noise_arrays = [np.random.rand(20, 20) for _ in range(8)]

# Step 2: Compute the LSE between each pair of arrays
def compute_lse(arr1, arr2):
    return np.sum((arr1 - arr2) ** 2)

# Step 3: Order the arrays by similarity based on the LSE
# Generate all permutations of the arrays
all_permutations = list(permutations(noise_arrays))

# Compute the total LSE for each permutation
lse_values = []
for perm in all_permutations:
    total_lse = 0
    for i in range(len(perm) - 1):
        total_lse += compute_lse(perm[i], perm[i + 1])
    lse_values.append(total_lse)

# Find the permutation with the minimum total LSE
min_lse_index = np.argmin(lse_values)
best_permutation = all_permutations[min_lse_index]

# Function to plot arrays in a grid
def plot_arrays(arrays, title):
    fig, axes = plt.subplots(1, 8, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(arrays[i], cmap='gray')
        ax.axis('off')
    fig.suptitle(title)
    plt.show()

# Plot the unordered arrays
plot_arrays(noise_arrays, "Unordered Arrays")

# Plot the ordered arrays
plot_arrays(best_permutation, "Ordered Arrays")