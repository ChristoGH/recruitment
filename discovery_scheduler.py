import pika
import time
import json
import threading
import logging
from datetime import datetime, timedelta
import os

# Set up logging
LOG_DIR = '/app/logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create file handlers
discovery_logger = logging.getLogger('discovery_service')
file_handler = logging.FileHandler(f'{LOG_DIR}/discovery_service.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
discovery_logger.addHandler(file_handler)

# Create separate scheduler log
scheduler_logger = logging.getLogger('discovery_scheduler')
scheduler_handler = logging.FileHandler(f'{LOG_DIR}/scheduler.log')
scheduler_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
scheduler_logger.addHandler(scheduler_handler)

QUEUE_NAME = 'url_queue'

def get_next_interval(interval_minutes=60):
    """Calculate the next closest interval time"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second
    current_microsecond = now.microsecond
    
    # Calculate minutes until next hour mark
    minutes_to_wait = interval_minutes - (current_minute % interval_minutes)
    
    # Calculate the exact next interval time
    next_time = now + timedelta(
        minutes=minutes_to_wait,
        seconds=-current_second,
        microseconds=-current_microsecond
    )
    
    return next_time

def connect_to_rabbitmq():
    credentials = pika.PlainCredentials(
        os.getenv('RABBITMQ_USER', 'guest'),
        os.getenv('RABBITMQ_PASSWORD', 'guest')
    )
    parameters = pika.ConnectionParameters(
        host=os.getenv('RABBITMQ_HOST', 'rabbitmq'),
        port=int(os.getenv('RABBITMQ_PORT', 5672)),
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300
    )
    return pika.BlockingConnection(parameters)

def get_queue_info(channel):
    """Get information about the queue including message count"""
    try:
        queue_info = channel.queue_declare(queue=QUEUE_NAME, durable=True, passive=True)
        message_count = queue_info.method.message_count
        discovery_logger.info(f"Current messages in queue: {message_count}")
        return message_count
    except Exception as e:
        discovery_logger.error(f"Error getting queue info: {e}")
        return 0

def publish_discovery_message(channel):
    discovery_message = {
        "timestamp": datetime.now().isoformat(),
        "batch_id": f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "type": "hourly_discovery"
    }
    
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=json.dumps(discovery_message),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
            headers={'source': 'discovery_scheduler'}
        )
    )
    message = f"Discovery scheduler triggered batch_id={discovery_message['batch_id']}"
    discovery_logger.info(message)
    scheduler_logger.info(message)

def scheduler():
    while True:
        try:
            connection = connect_to_rabbitmq()
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            
            while True:
                # Calculate next hour mark
                next_run = get_next_interval(60)
                
                # Sleep until exactly the next hour mark
                sleep_seconds = (next_run - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                
                # Publish discovery message at the interval time
                publish_discovery_message(channel)
                get_queue_info(channel)  # Log current queue status
                
        except Exception as e:
            discovery_logger.error(f"Scheduler error: {e}")
            time.sleep(5)
        finally:
            try:
                connection.close()
            except:
                pass

def main():
    try:
        # Initial connection test
        connection = connect_to_rabbitmq()
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        connection.close()
        
        discovery_logger.info("Starting URL discovery scheduler...")
        scheduler_logger.info("URL discovery scheduler initialized")
        
        # Start scheduler in main thread
        scheduler()
        
    except Exception as e:
        discovery_logger.error(f"Initialization error: {e}")
        raise

if __name__ == "__main__":
    main()