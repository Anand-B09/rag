#!/bin/bash

# Configurable variables
HOST="localhost"
PORT=11434
MODEL="gemma3:1b"

echo "Checking Ollama server at http://${HOST}:${PORT} ..."

# 1. Basic health check (test if server port is open)
echo "1. Testing if Ollama port is reachable..."
if nc -z "$HOST" "$PORT"; then
  echo "PASS: Ollama port ${PORT} is reachable."
else
  echo "FAIL: Cannot reach Ollama at ${HOST}:${PORT}"
  exit 1
fi

# 2. Test model list API
echo "2. Testing model list endpoint..."
models_output=$(curl -s "http://${HOST}:${PORT}/api/models" || echo "")
if [ -n "$models_output" ]; then
  echo "PASS: Model list retrieved successfully."
  echo "Available models: $models_output"
else
  echo "FAIL: Could not retrieve model list."
  exit 1
fi

# 3. Test a simple generation request with the given model
echo "3. Testing text generation API with model '${MODEL}'..."

# Example prompt
prompt="Hello Ollama, please reply briefly."

generate_json=$(cat << EOF
{
  "model": "${MODEL}",
  "prompt": "${prompt}",
  "max_tokens": 20,
  "stream": false
}
EOF
)

generate_response=$(curl -s -X POST "http://${HOST}:${PORT}/api/generate" \
  -H "Content-Type: application/json" \
  -d "$generate_json" || echo "")

if echo "$generate_response" | grep -q '"response"'; then
  echo "PASS: Text generation succeeded."
  echo "Response: $generate_response"
else
  echo "FAIL: Text generation failed or no 'response' field in response."
  echo "Response: $generate_response"
  exit 1
fi

echo "All tests passed. Ollama is working properly."

