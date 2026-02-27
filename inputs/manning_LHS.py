import numpy as np
from pyDOE import lhs

# Define the range for Manning values
manning_min = 0.0160
manning_max = 0.0360

# Define the number of samples
num_samples = 100  # Adjust this as needed

# Generate Latin Hypercube Sampling
lhs_samples = lhs(1, samples=num_samples, criterion='maximin')

# Map the Latin Hypercube Sampling to the Manning range
manning_samples = manning_min + lhs_samples * (manning_max - manning_min)

# Format Manning values to four decimal places and convert to strings
manning_samples_formatted = [float(format(manning, '.4f')) for manning in manning_samples.flatten()]

# Print as array
print("Latin Hypercube Sampling for Manning values:")
print(manning_samples_formatted)

# Write Manning values to a text file
output_file = "manning_samples_LHS.txt"
with open(output_file, "w") as f:
    for manning in manning_samples_formatted:
        f.write(f"{manning}\n")

print(f"Manning values extracted and saved to '{output_file}'")