#!/bin/bash
# Runs the model ensemble for Manning coefficient uncertainty quantification.
# Manning values are read from inputs/manning_samples_LHS.txt (Latin Hypercube Samples).
# For each sample: updates simulation_parameters.py, runs preprocessing -> run,
# then extracts R_mean and E_mean and appends to manning_results.txt.

set -euo pipefail

source ~/firedrake/bin/activate

samples_file="inputs/manning_samples_LHS.txt"
output_file="manning_results.txt"

if [[ ! -f "$samples_file" ]]; then
    echo "ERROR: samples file not found: $samples_file" >&2
    exit 1
fi

echo "Manning,R_mean,E_mean" > "$output_file"

start_time=$(date +%s)

while IFS= read -r manning; do
    echo "########################################################"
    echo "Manning = $manning"
    echo "########################################################"

    sed -i "s|manning_bkg = .*|manning_bkg = ${manning}|g" inputs/simulation_parameters.py

    mpirun.mpich -np 1 python preprocessing.py
    mpirun.mpich -np 6 python run.py

    read R_mean E_mean < <(python3 calculate_tidal_range_and_energy.py)
    echo "R_mean: $R_mean, E_mean: $E_mean"

    echo "${manning},${R_mean},${E_mean}" >> "$output_file"

done < "$samples_file"

end_time=$(date +%s)
duration=$((end_time - start_time))
total_hours=$(echo "scale=2; $duration / 3600" | bc)
echo "Total time: ${total_hours} hours"

deactivate
