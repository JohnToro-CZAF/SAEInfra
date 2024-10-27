from celery import Celery
from celery.schedules import crontab

# TASK_LIST = [
#     "jobapp.insert"
# ]

celery=None


def create_celery_app(app=None):
    celery = Celery(
        app.import_name,
        # backend=app.config['CELERY_RESULT_BACKEND'],
        # broker=app.config['CELERY_BROKER_URL'],
        # include=TASK_LIST
    )
    # celery.conf.update(app.config)
    celery.config_from_object(app.config["CELERY"])
    celery.set_default()
    # Schedule the gpu_manager task
    
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery

