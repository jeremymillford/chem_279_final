import matplotlib.pyplot as plt
import re
import numpy as np

# Parse the SCF output to extract iteration numbers and delta values
def parse_scf_data(scf_output):
    iteration_pattern = r"Iteration (\d+) delta: ([\d\.]+)"
    perturbation_pattern = r"Applied small perturbation at iteration (\d+)"
    
    iterations = []
    deltas = []
    perturbations = []
    
    for line in scf_output.split('\n'):
        # Extract iteration and delta
        match = re.search(iteration_pattern, line)
        if match:
            iterations.append(int(match.group(1)))
            deltas.append(float(match.group(2)))
        
        # Extract perturbation points
        match = re.search(perturbation_pattern, line)
        if match:
            perturbations.append(int(match.group(1)))
    
    return iterations, deltas, perturbations

# The SCF output
scf_output = """[CNDO/S] Using overlap approach with DIIS acceleration
Starting SCF iterations with fixed-damping DIIS...
Iteration 0 delta: 2.20746 (damping: 0.2, DIIS dim: 1, level shift: 0.01, error: 398.924)
Iteration 10 delta: 1.1534 (damping: 0.2, DIIS dim: 5, level shift: 0.11, error: 215.466)
Iteration 20 delta: 0.800103 (damping: 0.2, DIIS dim: 5, level shift: 0.21, error: 204.322)
Applied small perturbation at iteration 29
Iteration 30 delta: 1.28893 (damping: 0.2, DIIS dim: 5, level shift: 0.31, error: 229.592)
Iteration 40 delta: 0.792138 (damping: 0.2, DIIS dim: 5, level shift: 0.41, error: 187.893)
Iteration 50 delta: 0.448593 (damping: 0.2, DIIS dim: 5, level shift: 0.51, error: 218.659)
Applied small perturbation at iteration 59
Iteration 60 delta: 0.199123 (damping: 0.2, DIIS dim: 5, level shift: 0.61, error: 174.798)
Iteration 70 delta: 0.205128 (damping: 0.2, DIIS dim: 5, level shift: 0.71, error: 127.725)
Iteration 80 delta: 0.208886 (damping: 0.2, DIIS dim: 5, level shift: 0.81, error: 126.569)
Applied small perturbation at iteration 89
Iteration 90 delta: 0.297915 (damping: 0.2, DIIS dim: 5, level shift: 0.91, error: 125.217)
Iteration 100 delta: 0.100429 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 117.956)
Iteration 110 delta: 0.128458 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 115.147)
Applied small perturbation at iteration 119
Iteration 120 delta: 0.107055 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 119.384)
Iteration 130 delta: 0.235702 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 119.015)
Iteration 140 delta: 0.105667 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 123.127)
Applied small perturbation at iteration 149
Iteration 150 delta: 0.190448 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 116.15)
Iteration 160 delta: 0.180877 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 129.377)
Iteration 170 delta: 0.119458 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 117.471)
Applied small perturbation at iteration 179
Iteration 180 delta: 0.185701 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 125.879)
Iteration 190 delta: 0.613126 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 149.479)
Iteration 200 delta: 0.372412 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 138.932)
Applied small perturbation at iteration 209
Iteration 210 delta: 0.38619 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 138.216)
Iteration 220 delta: 0.270963 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 130.39)
Iteration 230 delta: 0.138654 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 122.377)
Applied small perturbation at iteration 239
Iteration 240 delta: 0.220273 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 130.468)
Iteration 250 delta: 0.124734 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 121.281)
Iteration 260 delta: 0.156675 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 136.115)
Applied small perturbation at iteration 269
Iteration 270 delta: 0.28611 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 144.543)
Iteration 280 delta: 0.156189 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 124.64)
Iteration 290 delta: 0.267498 (damping: 0.2, DIIS dim: 5, level shift: 1, error: 120.68)
Applied small perturbation at iteration 299
WARNING: SCF did not converge after 300 iterations.
Returning best approximate solution."""

# Parse the data
iterations, deltas, perturbations = parse_scf_data(scf_output)

# Find the minimum delta value and its iteration
min_delta = min(deltas)
min_delta_idx = deltas.index(min_delta)
min_delta_iteration = iterations[min_delta_idx]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot delta values
plt.plot(iterations, deltas, 'b-', label='Delta (Change in density matrix)')
plt.scatter(iterations, deltas, color='blue', s=30)

# Mark the minimum delta
plt.scatter(min_delta_iteration, min_delta, color='red', s=100, 
            label=f'Minimum delta: {min_delta:.6f} at iteration {min_delta_iteration}')

# Mark perturbation points
for p in perturbations:
    plt.axvline(x=p, color='r', linestyle='--', alpha=0.3)

# Add a horizontal line at the convergence threshold (assuming 1e-6)
convergence_threshold = 1e-6
plt.axhline(y=convergence_threshold, color='g', linestyle='--', 
            label=f'Typical convergence threshold (1e-6)')

# Customize the plot
plt.title('SCF Convergence Analysis', fontsize=16)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Delta (Change in density matrix)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.yscale('log')  # Use log scale for better visualization

# Add annotations explaining the perturbations
plt.annotate('Small perturbations applied', xy=(perturbations[0], 0.5), 
             xytext=(perturbations[0]-20, 0.8), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)

# Show best approximate solution details
plt.text(150, 3, f'Best approximate solution:\nDelta = {min_delta:.6f}\nIteration = {min_delta_iteration}',
         fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.show()