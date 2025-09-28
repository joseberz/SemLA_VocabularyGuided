#!/bin/bash

# Get configuration file as input
config_file="$1"

# Load directory paths from JSON
log_directory_parent=$(jq -r '.results_folder_parent' "$config_file")
log_directory_results="$log_directory_parent/$(jq -r '.experiment_name' "$config_file")"

# Create directories if they don't already exist
mkdir -p "$log_directory_parent"
mkdir -p "$log_directory_results"

# Define output and error file paths
output_file="$log_directory_results/output.txt"
error_file="$log_directory_results/error.txt"
export DETECTRON2_DATASETS=/leonardo_work/EUHPC_D11_069/datasets/

# Submit the job with dynamically set output and error paths, passing config file as an argument
sbatch --export=ALL \
       -A EUHPC_D11_069 \
       -p boost_usr_prod \
       -o "$output_file" \
       -e "$error_file" \
       --time=00:01:00 \
       -N 1 \
       --ntasks=1 \
       --cpus-per-task=4 \
       --gres=gpu:1 \
       --mem=30000 \
       --job-name=experiment \
       experiments_GL.sh "$config_file"
