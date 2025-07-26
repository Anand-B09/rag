#!/bin/bash

# Start Ollama server in the background
/bin/ollama serve &
pid=$!

# Wait a few seconds to allow server startup
sleep 5

# Pull the gemma3:1b model (only pulls if not already downloaded)
ollama pull gemma3:1b

# Wait for the Ollama server process to finish
wait $pid
