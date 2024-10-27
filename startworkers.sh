#!/bin/bash

# Define Celery application
CELERY_APP="jobapp.celery_app.celery_worker.celery"

# Define log directory
LOG_DIR="./celery_logs"
mkdir -p $LOG_DIR

# Function to start a Celery worker
start_worker() {
    QUEUE_NAME=$1
    WORKER_NAME=$2
    LOG_FILE="$LOG_DIR/${WORKER_NAME}.log"
    ERROR_FILE="$LOG_DIR/${WORKER_NAME}.err"

    echo "Starting worker '$WORKER_NAME' for queue '$QUEUE_NAME'..."
    nohup celery -A $CELERY_APP worker --loglevel=info -P threads -Q $QUEUE_NAME --hostname=$WORKER_NAME@%h > "$LOG_FILE" 2> "$ERROR_FILE" &
    echo "Worker '$WORKER_NAME' started with PID $!."
}

# Function to start the Celery beat scheduler
start_beat() {
    BEAT_NAME="celery-beat"
    LOG_FILE="$LOG_DIR/${BEAT_NAME}.log"
    ERROR_FILE="$LOG_DIR/${BEAT_NAME}.err"

    echo "Starting Celery beat..."
    nohup celery -A $CELERY_APP beat --loglevel=info --hostname=$BEAT_NAME@%h > "$LOG_FILE" 2> "$ERROR_FILE" &
    echo "Celery beat started with PID $!."
}

# Start Celery beat
start_beat

# Start Celery workers for specific queues
start_worker "queue1" "celery-worker-1"
start_worker "queue2" "celery-worker-2"
start_worker "queue3" "celery-worker-3"

echo "All Celery workers and beat scheduler have been started."