"""
Celery Configuration for Production-Ready Queue System
Phase 1: Core Infrastructure
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Create Celery app
celery_app = Celery('careai_tasks')

# Configuration
celery_app.conf.update(
    # Broker settings
    broker_url=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    
    # Serialization
    task_serializer=os.environ.get('CELERY_TASK_SERIALIZER', 'json'),
    result_serializer=os.environ.get('CELERY_RESULT_SERIALIZER', 'json'),
    accept_content=[os.environ.get('CELERY_ACCEPT_CONTENT', 'json').strip('[]"').replace('"', '')],
    
    # Timezone
    timezone=os.environ.get('CELERY_TIMEZONE', 'UTC'),
    
    # Task routing and concurrency
    task_routes={
        'tasks.process_file_upload': {'queue': 'file_uploads'},
        'tasks.process_embedding': {'queue': 'embeddings'},
        'tasks.send_webhook': {'queue': 'webhooks'},
        'tasks.process_agent_task': {'queue': 'agents'},
    },
    
    # Worker configuration
    worker_concurrency=int(os.environ.get('MAX_CONCURRENT_UPLOADS', 5)),
    worker_prefetch_multiplier=1,  # Important for long-running tasks
    
    # Task execution settings
    task_acks_late=True,  # Acknowledge task after completion
    worker_disable_rate_limits=False,
    task_default_retry_delay=int(os.environ.get('RETRY_DELAY', 5)),
    task_max_retries=int(os.environ.get('RETRY_MAX_ATTEMPTS', 3)),
    task_time_limit=int(os.environ.get('TASK_TIMEOUT', 1800)),  # 30 minutes
    task_soft_time_limit=int(os.environ.get('TASK_TIMEOUT', 1800)) - 60,  # 29 minutes
    
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Import tasks to register them
from tasks import *