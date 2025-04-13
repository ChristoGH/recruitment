#!/bin/bash

# Install required packages
apt-get update && apt-get install -y cron curl

# Create logs directory if it doesn't exist
mkdir -p /app/logs

# Set up cron job to trigger the search endpoint instead of starting a new service
echo '0 * * * * curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d "{\"id\":\"batch1\",\"days_back\":7}" >> /app/logs/cron.log 2>&1' > /etc/cron.d/discovery-cron
chmod 0644 /etc/cron.d/discovery-cron
crontab /etc/cron.d/discovery-cron

# Start the service in the background
/usr/local/bin/python -m uvicorn url_discovery_service:app --host 0.0.0.0 --port 8000 > /app/logs/startup.log 2>&1 &

# Wait for service to start
sleep 10

# Trigger initial search
curl -X POST 'http://localhost:8000/search' -H 'Content-Type: application/json' -d '{"id":"batch1","days_back":7}' > /app/logs/initial_run.log 2>&1

# Keep the container running and run cron in the foreground
cron -f 