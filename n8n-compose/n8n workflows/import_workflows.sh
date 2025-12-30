#!/bin/bash

# --- CONFIGURATION ---
N8N_DOMAIN="primary-production-77a53.up.railway.app"
N8N_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzNGRlNWQ4Yy1lMGY5LTQ3YzctYjkxZS01ZDc4MzFjYTUwM2IiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzU3MzgzNDYxLCJleHAiOjE3NTk4OTYwMDB9.wMaTQ6ynhRQv-rjITk9kwqCXCzfEH1ifsBgYn3uOGAg"
# --- END CONFIGURATION ---

# Check if curl and jq are installed
if ! command -v curl &>/dev/null || ! command -v jq &>/dev/null; then
  echo "Error: Please ensure both 'curl' and 'jq' are installed."
  echo "On Debian/Ubuntu, run: sudo apt-get install curl jq"
  exit 1
fi

# Loop through all .json files
for file in *.json; do
  [ -f "$file" ] || continue

  echo "--------------------------------"
  echo "Processing workflow: $file ..."

    # Validate JSON
    if ! jq -e . >/dev/null 2>&1 < "$file"; then
        echo "Error: '$file' is not a valid JSON file. Skipping."
        continue
    fi
    # Skip array-style exports
    if jq -e 'type == "array"' "$file" >/dev/null; then
    echo "Skipping '$file' (array, not a workflow export)."
    continue
    fi
  # Remove 'id' field and send to API
    response=$(jq '{
    name: .name,
    nodes: .nodes,
    connections: .connections,
    settings: .settings,
    }' "$file" | curl -s -w "\n%{http_code}" -X POST \
    "https://${N8N_DOMAIN}/api/v1/workflows" \
    -H "X-N8N-API-KEY: ${N8N_API_KEY}" \
    -H "Content-Type: application/json" \
    --data-binary @-)


    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" -eq 201 ]; then
        echo "Successfully imported '$file' (Status: $http_code)."
    else
        echo "Error importing '$file' (Status: $http_code)."
        echo "API Response: $body"
    fi
done

echo "--------------------------------"
echo "All workflows have been processed."
