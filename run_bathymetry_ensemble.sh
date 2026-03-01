#!/bin/bash
# Runs the model ensemble for bathymetric uncertainty quantification.
# bath_error values are read from inputs/bath_samples_LHS.txt (Latin Hypercube Samples).
# For each sample: updates simulation_parameters.py, then runs preprocessing -> ramp -> run.

set -euo pipefail

source ~/firedrake/bin/activate

samples_file="inputs/bath_samples_LHS.txt"

if [[ ! -f "$samples_file" ]]; then
    echo "ERROR: samples file not found: $samples_file" >&2
    exit 1
fi

while IFS= read -r bath_error; do
    echo "########################################################"
    echo "Bath error = $bath_error"
    echo "########################################################"

    start_time=$(date +%s)

    current_output="outputs/outputs_run/H=${bath_error}"
    mkdir -p "$current_output"

    sed -i "s|bath_error = .*|bath_error = ${bath_error}|g" inputs/simulation_parameters.py
    sed -i "s|run_output_folder = .*|run_output_folder = '${current_output}'|g" inputs/simulation_parameters.py

    mpirun.mpich -np 1 python preprocessing.py
    mpirun.mpich -np 6 python ramp.py
    mpirun.mpich -np 6 python run.py

    end_time=$(date +%s)
    duration=$((end_time - start_time))
    total_hours=$(echo "scale=2; $duration / 3600" | bc)
    echo "Time taken for bath_error=${bath_error}: ${total_hours} hours"

done < "$samples_file"

deactivate
