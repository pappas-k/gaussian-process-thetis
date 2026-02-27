import numpy as np
from pyDOE import lhs

# Define the range for Manning values
bath_error_min = -3.00
bath_error_max = +3.00

# Define the number of samples
num_samples = 100  # Adjust this as needed

# Generate Latin Hypercube Sampling
lhs_samples = lhs(1, samples=num_samples, criterion='maximin')

# Map the Latin Hypercube Sampling to the Manning range
bath_error_samples = bath_error_min + lhs_samples * (bath_error_max - bath_error_min)

# Format Manning values to four decimal places and convert to strings
bath_error_samples_formatted = [float(format(error, '.2f')) for error in bath_error_samples.flatten()]

# Print as array
print("Latin Hypercube Sampling for Bathymetric values:")
print(bath_error_samples_formatted)

# Write Manning values to a text file
output_file = "bath_samples_LHS.txt"
with open(output_file, "w") as f:
    for manning in bath_error_samples_formatted:
        f.write(f"{manning}\n")

print(f"Manning values extracted and saved to '{output_file}'")