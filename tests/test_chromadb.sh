output=$(curl -s http://localhost:9000/api/v2/healthcheck)

expected='{"is_executor_ready":true,"is_log_client_ready":true}'

if [ "$output" = "$expected" ]; then
  echo "PASS: Healthcheck output matches expected."
else
  echo "FAIL: Healthcheck output does not match expected."
  echo "Actual output: $output"
fi