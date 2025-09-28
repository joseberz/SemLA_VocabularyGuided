source_folder="experiments_config_files/SED_jan_20"
results_folder="results/SED_jan_20"
mkdir -p "$results_folder"

export DETECTRON2_DATASETS=/leonardo_work/EUHPC_D11_069/datasets/

for folder in "$source_folder"/*; do
    config_file_folder="$folder"
    folder_name=$(basename "$folder")
    results_folder_parent="$results_folder/$folder_name"
    for config_file in "$config_file_folder"/*.json; do
        echo "Submitting job for configuration file: $config_file"
        # Load directory paths from JSON
        experiment_name=$(jq -r '.experiment_name' "$config_file")
        log_directory_results="$results_folder_parent/$experiment_name"
        output_file="$log_directory_results/output.txt"
        error_file="$log_directory_results/error.txt"
        echo "Results folder: $log_directory_results"
        mkdir -p "$log_directory_results"
        sbatch --export=ALL \
        -A EUHPC_D11_069 \
        -p boost_usr_prod \
        -o "$output_file" \
        -e "$error_file" \
        --time=23:00:00 \
        -N 1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --gres=gpu:1 \
        --mem=60000 \
        --job-name=table_gl \
        experiments_GL.sh "$config_file" "$log_directory_results"        
done


done

