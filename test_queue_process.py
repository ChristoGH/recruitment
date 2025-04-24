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
test_logger = logging.getLogger('test_service')
file_handler = logging.FileHandler(f'{LOG_DIR}/test_service.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
test_logger.addHandler(file_handler)

# Create separate cron simulation log
cron_logger = logging.getLogger('test_cron')
cron_handler = logging.FileHandler(f'{LOG_DIR}/cron.log')
cron_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
cron_logger.addHandler(cron_handler)

QUEUE_NAME = 'url_queue'

def get_next_interval(interval_minutes=15):
    """Calculate the next closest interval time"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second
    current_microsecond = now.microsecond
    
    # Calculate minutes until next 15-minute mark
    minutes_to_wait = interval_minutes - (current_minute % interval_minutes)
    
    # Calculate the exact next interval time
    next_time = now + timedelta(
        minutes=minutes_to_wait,
        seconds=-current_second,
        microseconds=-current_microsecond
    )
    
    return next_time

def connect_to_rabbitmq():
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(
        host='rabbitmq',
        port=5672,
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
        test_logger.info(f"Current messages in queue: {message_count}")
        return message_count
    except Exception as e:
        test_logger.error(f"Error getting queue info: {e}")
        return 0

def purge_queue(channel, queue_name):
    """Purge all messages from the queue"""
    try:
        message_count = channel.queue_purge(queue=queue_name).method.message_count
        test_logger.info(f"Purged {message_count} messages from queue")
    except Exception as e:
        test_logger.error(f"Error purging queue: {e}")

def publish_test_message(channel):
    test_message = {
        "url": f"https://test.com/{datetime.now().isoformat()}",
        "timestamp": datetime.now().isoformat(),
        "test_batch": True
    }
    
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=json.dumps(test_message),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
            headers={'source': 'test_producer'}
        )
    )
    message = f"Test producer triggered batch_id=test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_logger.info(message)
    cron_logger.info(message)

def producer():
    while True:
        try:
            connection = connect_to_rabbitmq()
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            
            while True:
                # Calculate next 15-minute mark
                next_run = get_next_interval(15)
                
                # Sleep until exactly the next 15-minute mark
                sleep_seconds = (next_run - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                
                # Publish message at the interval time
                publish_test_message(channel)
                get_queue_info(channel)  # Log current queue status
                
        except Exception as e:
            test_logger.error(f"Producer error: {e}")
            time.sleep(5)
        finally:
            try:
                connection.close()
            except:
                pass

def process_message(message_body):
    """Process a single message"""
    try:
        message_data = json.loads(message_body)
        message_timestamp = datetime.fromisoformat(message_data['timestamp'])
        age = datetime.now() - message_timestamp
        
        test_logger.info(f"Processing message (age: {age.total_seconds():.1f}s): {message_data}")
        
        if age.total_seconds() > 3600:  # older than 1 hour
            test_logger.warning(f"Message is {age.total_seconds()/3600:.2f} hours old")
            
        return True
    except Exception as e:
        test_logger.error(f"Error processing message: {e}")
        return False

def callback(ch, method, properties, body):
    try:
        success = process_message(body.decode())
        if success:
            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    except Exception as e:
        test_logger.error(f"Callback error: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def consumer():
    while True:
        try:
            connection = connect_to_rabbitmq()
            channel = connection.channel()
            
            queue_info = channel.queue_declare(queue=QUEUE_NAME, durable=True)
            initial_message_count = queue_info.method.message_count
            test_logger.info(f"Consumer starting. Found {initial_message_count} messages in queue")
            
            # Fair dispatch
            channel.basic_qos(prefetch_count=1)
            
            channel.basic_consume(
                queue=QUEUE_NAME,
                on_message_callback=callback
            )
            
            test_logger.info("Started consuming messages...")
            channel.start_consuming()
            
        except Exception as e:
            test_logger.error(f"Consumer error: {e}")
            time.sleep(5)
        finally:
            try:
                connection.close()
            except:
                pass

def delayed_consumer_start():
    """Wait 5 minutes before starting the consumer"""
    delay_minutes = 5
    test_logger.info(f"Consumer will start in {delay_minutes} minutes. Check RabbitMQ UI at http://localhost:15672 (guest/guest)")
    time.sleep(delay_minutes * 60)  # Convert to seconds
    test_logger.info("Starting consumer after delay...")
    consumer()

def main():
    try:
        connection = connect_to_rabbitmq()
        channel = connection.channel()
        
        # Check if queue exists and get message count
        queue_info = channel.queue_declare(queue=QUEUE_NAME, durable=True, passive=True)
        message_count = queue_info.method.message_count
        
        if message_count > 0:
            test_logger.warning(f"Found {message_count} existing messages in queue")
            user_input = input("Do you want to purge existing messages before starting? (yes/no): ")
            if user_input.lower() == 'yes':
                purge_queue(channel, QUEUE_NAME)
        
        connection.close()
    except Exception as e:
        test_logger.error(f"Initial queue check failed: {e}")

    # Start producer and delayed consumer in separate threads
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=delayed_consumer_start)
    
    producer_thread.start()
    consumer_thread.start()
    
    # Wait for both threads
    producer_thread.join()
    consumer_thread.join()

if __name__ == "__main__":
    main() 