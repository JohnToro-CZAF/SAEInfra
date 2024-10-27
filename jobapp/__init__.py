# app/__init__.py
import time
from flask import Flask, g
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from kombu import Exchange, Queue

from jobapp.config import Config
from jobapp import celery_app


db = SQLAlchemy()

def create_app():
    """Create a Flask application.
    """
    # Instantiate Flask
    app = Flask(__name__)
    CORS(app)
    crsf = CSRFProtect(app)
    crsf.init_app(app)

    # Load common settings
    app.config.from_object(Config)
    app.config.from_mapping(
        CELERY=dict(
            broker_url=app.config['CELERY_BROKER_URL'],
            result_backend=app.config['CELERY_RESULT_BACKEND'],
            beat_schedule={
                'gpu-manager-every-5-seconds': {
                    'task': 'jobapp.gpu_manager',
                    'schedule': 5,  # seconds
                }
            },
            # Define task queues
            task_queues = (
                Queue('gpu0_0_queue', Exchange('gpu0_0_queue'), routing_key='gpu0_0_queue'),
                Queue('gpu0_1_queue', Exchange('gpu0_1_queue'), routing_key='gpu0_1_queue'),
                # Queue('gpu0_2_queue', Exchange('gpu0_2_queue'), routing_key='gpu0_2_queue'),
                # Queue('gpu0_3_queue', Exchange('gpu0_3_queue'), routing_key='gpu0_3_queue'),
                # Queue('gpu1_0_queue', Exchange('gpu1_0_queue'), routing_key='gpu1_0_queue'),
                # Queue('gpu1_1_queue', Exchange('gpu1_1_queue'), routing_key='gpu1_1_queue'),
                # Queue('gpu1_2_queue', Exchange('gpu1_2_queue'), routing_key='gpu1_2_queue'),
                # Queue('gpu1_3_queue', Exchange('gpu1_3_queue'), routing_key='gpu1_3_queue'),
                # Add more queues if needed
                Queue("default", Exchange("default"), routing_key="default")
            ),
            task_routes = {
                'jobapp.gpu_manager': {'queue': 'default'},
            }

            # Optionally, set a default queue
            # task_default_queue = 'default_queue',
            # task_default_exchange = 'default_queue',
            # task_default_routing_key = 'default_queue',
        )
    )
    # Setup Flask-SQLAlchemy
    db.init_app(app)

    # Celery
    celery = celery_app.create_celery_app(app)
    celery_app.celery = celery

    # Register blueprints
    from jobapp.routes import insert_bp, task_bp
    app.register_blueprint(insert_bp)
    app.register_blueprint(task_bp)
    return app