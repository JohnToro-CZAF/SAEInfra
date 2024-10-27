# app/routes/jobs.py

import os
import uuid
from flask import Blueprint, request, jsonify, send_file, render_template
from flask_restful import Api, Resource

from flask import (
    Blueprint, 
    request,
    send_from_directory
)

from jobapp.models import Result, Job
from jobapp.tasks import insert
from jobapp.tasks import jobtask, longtask, visiontask
from jobapp import db
from jobapp.utils import get_system_resources

insert_bp = Blueprint('insert', __name__)

@insert_bp.route('/insertData')
def insertData():
    insert.delay()
    return "Data inserted!"

task_bp = Blueprint('authentication', __name__)

@task_bp.route('/create_job', methods=['POST'])
def create_job():
#     print("coming:", request.get_json())
#     data = request.get_json()
#     model_name = data.get('model_name')
#     dataset_name = data.get('dataset_name')

#     if not model_name or not dataset_name:
#         return {'message': 'Model name and Dataset name are required.'}, 400

#     job_id = str(uuid.uuid4())
#     checkpoint_dir = 'checkpoints'
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     checkpoint_path = os.path.join(checkpoint_dir, f"{job_id}.pth")

#     new_job = Job(
#         job_id=job_id,
#         model_name=model_name,
#         dataset_name=dataset_name,
#         checkpoint_path=checkpoint_path,
#         status='stopped',
#         stage_status='created'
#     )
#     db.session.add(new_job)
#     db.session.commit()

#     return {'message': 'Job created', 'job_id': job_id}, 201
    data = request.get_json()
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    memory_required = data.get('memory_required', 1024)  # Default to 1GB if not specified
    prefered_gpu_id = data.get('gpu_id')  # GPU Device ID
    batch_size = data.get('batch_size', 32)  # Default to 32 if not specified

    if not model_name or not dataset_name or prefered_gpu_id is None or memory_required is None:
        return {'message': 'Model name, Dataset name, GPU ID, and Memory required are all required.'}, 400

    # Validate GPU ID
    if not isinstance(prefered_gpu_id, int):
        return {'message': 'GPU ID must be a integer.'}, 400

    if prefered_gpu_id < 0:
        prefered_gpu_id = -1
    
    # Optionally, validate memory_required
    if not isinstance(memory_required, int) or memory_required < 256:
        return {'message': 'Memory required must be an integer of at least 256 MB.'}, 400

    job_id = str(uuid.uuid4())
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{job_id}.pth")

    new_job = Job(
        job_id=job_id,
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        status='stopped',  # Set to 'queued' to allow Manager to assign GPU
        stage_status='created',
        memory_required=memory_required,
        running_memory=None,
        assigned_gpu=None,# Initially not assigned
        prefered_gpu=None if prefered_gpu_id == -1 else prefered_gpu_id, # Set to None if no preference, wait until GPU is free, then assign to that GPU,
        batch_size=batch_size
    )
    db.session.add(new_job)
    db.session.commit()

    return {'message': 'Job created', 'job_id': job_id}, 201

# @task_bp.route('/jobs', methods=['GET'])
# def get_jobs():
#     jobs = Job.query.all()
#     jobs_data = []
#     for job in jobs:
#         jobs_data.append({
#             'job_id': job.job_id,
#             'model_name': job.model_name,
#             'dataset_name': job.dataset_name,
#             'status': job.status,
#             'stage_status': job.stage_status,
#             'progress': job.progress,
#             "loss_history": job.get_loss_history(),
#             'created_at': job.created_at.isoformat() if job.created_at else None,
#             'updated_at': job.updated_at.isoformat() if job.updated_at else None
#         })
#     return {'jobs': jobs_data}, 200

@task_bp.route('/jobs', methods=['GET'])
def get_jobs():
    jobs = Job.query.all()
    jobs_data = []
    for job in jobs:
        jobs_data.append({
            'job_id': job.job_id,
            'model_name': job.model_name,
            'dataset_name': job.dataset_name,
            'status': job.status,
            'stage_status': job.stage_status,
            'progress': job.progress,
            "loss_history": job.get_loss_history(),
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'updated_at': job.updated_at.isoformat() if job.updated_at else None,
            'assigned_gpu': job.assigned_gpu,
            'prefered_gpu': job.prefered_gpu,
            "memory_required": job.memory_required,
            "running_memory": job.running_memory,
            "failure_reason": job.failure_reason
        })
    return {'jobs': jobs_data}, 200

@task_bp.route('/jobs/<string:job_id>/stop', methods=['POST'])
def stop_job(job_id):
    job = Job.query.filter_by(job_id=job_id).first()
    if not job:
        return {'message': 'Job not found'}, 404
    if job.status not in ['running', 'queued']:
        return {'message': f'Cannot stop a job that is {job.status}'}, 400
    job.status = 'stopped'
    db.session.commit()
    return {'message': 'Job stopped'}, 200

@task_bp.route('/jobs/<string:job_id>/continue', methods=['POST'])
def continue_job(job_id):
    job = Job.query.filter_by(job_id=job_id).first()
    if not job:
        return {'message': 'Job not found'}, 404
    if job.status != 'stopped':
        return {'message': f'Cannot continue a job that is {job.status}'}, 400
    job.status = 'queued'
    job.stage_status = 'queued'
    db.session.commit()
    
    # potentially resume the job once the task is allocated
    # visiontask.delay(job_id)
    return {'message': 'Job continued'}, 200

@task_bp.route('/api/system_resources', methods=['GET'])
def system_resources():
    resources = get_system_resources()
    return jsonify(resources)

@task_bp.route('/')
def serve_frontend():
    # return send_from_directory('static', 'index.html')
    return render_template('index.html')
