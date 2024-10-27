# app/models.py
import json
from jobapp import db
from datetime import datetime
    
class Result(db.Model):
    __tablename__ = 'result'
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(200), nullable=False)
    

class Job(db.Model):
    __tablename__ = 'job'
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(36), unique=True, nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    dataset_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='queued')  # queued, running, stopped, completed
    progress = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.now())
    updated_at = db.Column(db.DateTime, default=datetime.now(), onupdate=datetime.now())
    checkpoint_path = db.Column(db.String(200), nullable=True)
    loss_history = db.Column(db.Text, nullable=True)  # JSON-serialized list of loss values
    stage_status = db.Column(db.String(20), default='queued')  # created, queued, allocating, model_loading, data_loading, training, completed
    assigned_gpu = db.Column(db.Integer, nullable=True)  # GPU ID (e.g., 0, 1, 2...)
    prefered_gpu = db.Column(db.Integer, nullable=True)  # Prefered GPU ID (e.g., 0, 1, 2...)
    memory_required = db.Column(db.Integer, nullable=True)  # Memory in MB
    running_memory = db.Column(db.Integer, nullable=True)  # Peak GPU memory used during training (MB)
    failure_reason = db.Column(db.String(200), nullable=True)  # Reason for failure, if any
    batch_size = db.Column(db.Integer, nullable=True)
    
    def get_loss_history(self):
        if self.loss_history:
            return json.loads(self.loss_history)
        return []

    def add_loss(self, loss_value):
        history = self.get_loss_history()
        history.append(loss_value)
        self.loss_history = json.dumps(history)