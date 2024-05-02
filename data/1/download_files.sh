#!/bin/bash

# Path to the text file containing links
links_file="1.txt"

# Check if the links file exists
if [ ! -f "$links_file" ]; then
    echo "Error: Links file not found: $links_file"
    exit 1
fi

# Loop through each link in the file and download the file
while IFS= read -r link; do
    wget "$link"
done < "$links_file"
