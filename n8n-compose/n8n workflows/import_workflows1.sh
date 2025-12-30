#!/bin/bash

# --- CONFIGURATION ---
N8N_DOMAIN="n8n-production-9322.up.railway.app"
# !!! IMPORTANT !!!
# Make sure this is your NEW, valid API key.
N8N_API_KEY="YOUR_NEW_API_KEY_HERE"
# --- END CONFIGURATION ---

# Check if curl and jq are installed
if ! command -v curl &> /dev/null || ! command -v jq &> /dev/null; then
    echo "Error: Please ensure both 'curl' and 'jq' are installed."
    echo "On Debian/Ubuntu, run: sudo apt-get install curl jq"
    exit 1
fi

# Loop through all .json files
for file in *.json; do
    if [ -f "$file" ]; then
        echo "--------------------------------"
        echo "Processing workflow: $file ..."

        # Check if the file contains valid JSON
        if ! jq -e . >/dev/null 2>&1 < "$file"; then
            echo "Error: '$file' is not a valid JSON file. Skipping."
            continue
        fi

        # Remove top-level 'id' field and send to API
        response=$(jq 'del(.id)' "$file" | curl -s -w "\n%{http_code}" -X POST \
            "https://${N8N_DOMAIN}/api/v1/workflows" \
            -H "X-N8N-API-KEY: ${N8N_API_KEY}" \
            -H "Content-Type: application/json" \
            --data-binary @-)

        # Extract HTTP status code and body
        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | sed '$d')

        if [ "$http_code" -eq 201 ]; then
            echo "Successfully imported '$file' (Status: $http_code)."
        else
            echo "Error importing '$file' (Status: $http_code)."
            echo "API Response: $body"
        fi
    fi
done

echo "--------------------------------"
echo "All workflows have been processed."
