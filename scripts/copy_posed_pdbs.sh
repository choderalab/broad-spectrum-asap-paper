#!/bin/bash

# Define the main directory
if [ $# -eq 0 ];then
    echo "$1: Missing source directory"
    exit 1
    fi

main_dir="$1/docking_results" 
final_dir=$2
pattern=$3

mkdir $final_dir
# Loop over each item in the main directory
for dir in "$main_dir"/*; do
    if [ -d "$dir" ]; then
        subdir_name=$(basename "$dir")
	file_name=$(echo "$subdir_name" | grep -o -E "$pattern")
        echo file ${file_name}
        source_file="$dir/docked_complex_0.pdb"
        if [ -f "$source_file" ]; then
            destination_file="$final_dir/${file_name}_docked.pdb"
            cp "$source_file" "$destination_file"
            echo "Copied $source_file to $destination_file"
        else
            echo "'docked_complex_0.pdb' not found in $dir"
        fi
    fi
done
