from jobapp import create_app
from jobapp import celery_app

app = create_app()
celery = celery_app.create_celery_app(app)
celery_app.celery = celery