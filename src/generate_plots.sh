#!/bin/bash

echo "Starting execution of visualization scripts..."

# Enable nullglob so the loop handles empty matches gracefully
shopt -s nullglob

# Find all Python files starting with visualize_ in the current directory
SCRIPTS=(visualize_*.py)

# Check if any files were found
if [ ${#SCRIPTS[@]} -eq 0 ]; then
    echo "No matching files found in the current directory."
    exit 0
fi

for script in "${SCRIPTS[@]}"; do
    echo "Running $script"

    # Execute the python script
    python3 "$script"

    # Check the exit status of the python command
    if [[ $? -eq 0 ]]; then
        echo "Successfully finished $script"
    else
        echo "Error encountered in $script"
    fi
    echo ""
done

echo "All scripts have been processed."