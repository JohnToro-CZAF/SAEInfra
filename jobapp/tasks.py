# app/tasks.py
import os
import gc
import multiprocessing  
import time
import json
import tqdm
import random
import celery
import threading

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from jobapp.config import Config
from jobapp.models import Result, Job

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI_CELERY)
class GpuTask(celery.Task):
    def __init__(self):
        self.sessions = {}
        self.stop_event = None
        self.monitor_thread = None

    def before_start(self, task_id, args, kwargs):
        self.sessions[task_id] = Session(engine)
        job_id, gpu_id = args
        session = self.sessions[task_id]
        # peak_memory = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)
        job = session.query(Job).filter_by(job_id=job_id).first()
        self.stop_event = threading.Event()
        print("Starting to monitor memory for job: ", job_id)
        def monitoring_memory(job_id):
            ses = Session(engine)
            peak_memory = 0
            while not self.stop_event.is_set():
                current_memory = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)  # Convert to MB
                if current_memory > peak_memory:
                    peak_memory = current_memory
                print(f"Peak memory usage: {peak_memory} MB")
                ses.query(Job).filter_by(job_id=job_id).update({'running_memory': int(peak_memory)})
                ses.commit()
                time.sleep(1)
            ses.close()
        
        self.monitor_thread = threading.Thread(target=monitoring_memory, args=[job_id])
        self.monitor_thread.start()
        super().before_start(task_id, args, kwargs)

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        print("The task is done: ", task_id)
        print("Args: ", args)
        print("Status: ", status, type(status), str(status))
        print("Retval: ", retval, type(retval), str(retval))
        print("Einfo: ", einfo, type(einfo))
        print("Kwargs: ", kwargs)
        job_id, gpu_id = args
        session = self.sessions.pop(task_id)
        if str(status) == "FAILURE":
            job = session.query(Job).filter_by(job_id=job_id).first()
            if job:
                print("Job is not None, we enter here")
                job.status = 'failed'
                job.stage_status = 'Error during executing'
                job.failure_reason = str(retval)
                print(f"Job {job_id}: Job failed")
                session.commit()
        job = session.query(Job).filter_by(job_id=job_id).first()
        if job:
            job.assigned_gpu = None
            session.commit()
        torch.cuda.empty_cache()
        # clear all the memory
        gc.collect()
        
        self.stop_event.set()
        self.monitor_thread.join()
        torch.cuda.empty_cache() # clear all the memory
        gc.collect() 
        session.close()
        super().after_return(status, retval, task_id, args, kwargs, einfo)
    
    @property
    def session(self):
        return self.sessions[self.request.id]

from jobapp.decorators import record_peak_gpu_memory

from jobapp import db
from jobapp.celery_app import celery
from jobapp.utils import get_gpu_memory

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models


@celery.task(name="jobapp.insert",ignore_result=False)
def insert():
    for i in range(5):
        time.sleep(1)
        data = ''.join(random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']) for i in range(10))
        result = Result(data=data)
        db.session.add(result)
    db.session.commit()
    return "Data inserted!"

@celery.task(name="jobapp.jobtask",ignore_result=False)
def jobtask(job_id):
    job = Job.query.filter_by(job_id=job_id).first()
    if not job:
        print(f"Job {job_id} not found.")
        return
    
    job.status = 'running'
    db.session.commit()

    # Simulate loading a checkpoint if exists
    t = 0
    if job.checkpoint_path and os.path.exists(job.checkpoint_path):
        try:
            with open(job.checkpoint_path, 'r') as f:
                data = json.load(f)
            t = data.get('t', 0)
            print(f"Loaded checkpoint for job {job_id} at step {t}")
        except Exception as e:
            print(f"Failed to load checkpoint for job {job_id}: {e}")
            t = 0

    total_steps = 10000  # Simulate 10 steps
    for step in range(t, total_steps):
        if step % 10 == 0: # 0.01 * 100 = 1s to check status
            job = Job.query.filter_by(job_id=job_id).first()
            if job.status == 'stopped':
                # Save checkpoint
                try:
                    with open(job.checkpoint_path, 'w') as f:
                        json.dump({'t': step}, f)
                    print(f"Saved checkpoint for job {job_id} at step {step}")
                except Exception as e:
                    print(f"Failed to save checkpoint for job {job_id}: {e}")
                return

        # Simulate a step taking some time
        time.sleep(0.01)  # Simulate long-running task

        # Update progress
        job.progress = ((step + 1) / total_steps) * 100
        db.session.commit()
        print(f"Job {job_id} progress: {job.progress}%")

    # Mark job as completed
    job.status = 'completed'
    job.progress = 100.0
    db.session.commit()
    print(f"Job {job_id} completed.")

@celery.task(name="jobapp.modeltask",ignore_result=False)
def longtask(job_id):
    job = Job.query.filter_by(job_id=job_id).first()
    if not job:
        print(f"Job {job_id} not found.")
        return
    
    job.status = 'running'
    db.session.commit()
    
    # Example: Simple training loop
    # Replace with actual model and dataset
    db.session.commit()
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Simulate loading a checkpoint if exists
    start_epoch = 0
    if job.checkpoint_path and os.path.exists(job.checkpoint_path):
        try:
            checkpoint = torch.load(job.checkpoint_path)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
        except:
            start_epoch = 0

    total_epochs = 100  # Simulate 10 steps
    for epoch in range(start_epoch, total_epochs):
        if epoch % 1 == 0: # 0.01 * 100 = 1s to check status
            job = Job.query.filter_by(job_id=job_id).first()
            if job.status == 'stopped':
                # Save checkpoint
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, job.checkpoint_path or f'checkpoints/{job.job_id}.pth')
                    print(f"Saved checkpoint for job {job_id} at epoch {epoch}")
                except Exception as e:
                    print(f"Failed to save checkpoint for job {job_id}: {e}")
                return
        
        # Dummy training step
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        time.sleep(0.01)

        # Update progress
        job.progress = ((epoch + 1) / total_epochs) * 100
        db.session.commit()
        print(f"Job {job_id} progress: {job.progress}%")

    # Mark job as completed
    job.status = 'completed'
    job.progress = 100.0
    db.session.commit()
    print(f"Job {job_id} completed.")

@celery.task(bind=True, name="jobapp.visiontask", ignore_result=False, base=GpuTask)
# @record_peak_gpu_memory
def visiontask(self, job_id, gpu_id):
    # job = Job.query.filter_by(job_id=job_id).first()
    print(self.session)
    job = self.session.query(Job).filter_by(job_id=job_id).first()
    if not job:
        print(f"Job {job_id} not found.")
        return
    
    job.status = 'running'
    self.session.commit()
    
    job.stage_status = 'model_loading'
    self.session.commit()
    # Define device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # Load a pretrained model (e.g., ResNet18)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Modify the final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])
    
    job.stage_status = 'data_loading'
    self.session.commit()
    bs = job.batch_size
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='data', train=True,
                                     download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
                                               shuffle=True, num_workers=16)

    # Simulate loading a checkpoint if exists
    start_epoch = 0
    start_batch = 0
    if job.checkpoint_path and os.path.exists(job.checkpoint_path):
        try:
            checkpoint = torch.load(job.checkpoint_path)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
        except:
            start_epoch = 0
            start_batch = 0

    job.stage_status = 'training'
    self.session.commit()
    
    total_epochs = 40  # Simulate 10 steps
    for epoch in range(start_epoch, total_epochs):
        if epoch % 1 == 0: # 0.01 * 100 = 1s to check status
            job = self.session.query(Job).filter_by(job_id=job_id).first()
            # job = Job.query.filter_by(job_id=job_id).first()
            if job.status == 'stopped':
                # Save checkpoint
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, job.checkpoint_path or f'checkpoints/{job.job_id}.pth')
                    print(f"Saved checkpoint for job {job_id} at epoch {epoch}")
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Failed to save checkpoint for job {job_id}: {e}")
                return
        model.train()
        running_loss = 0.0
        cnt = 0
        for batch_idx, (inputs, labels) in tqdm.tqdm(enumerate(train_loader)):
            if batch_idx < start_batch:
                continue
            
            cnt += 1
            # Periodically check job status
            if batch_idx % 10 == 0:  # Adjust frequency as needed
                job = self.session.query(Job).filter_by(job_id=job_id).first()
                # job = Job.query.filter_by(job_id=job_id).first()
                if job.status == 'stopped':
                    # Save checkpoint before exiting
                    try:
                        torch.save({
                            'epoch': epoch,
                            'batch': batch_idx,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                        }, job.checkpoint_path or f'checkpoints/{job.job_id}.pth')
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Failed to save checkpoint for job {job_id}: {e}")
                    return  # Exit the training gracefully
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * inputs.size(0)
            if batch_idx % 10 == 0:
                current_epoch_loss = running_loss / (cnt * train_loader.batch_size)
                job.add_loss(current_epoch_loss)  # Append loss to loss_history
                job.progress = ((epoch + 1)/ total_epochs) * 100 + (batch_idx / len(train_loader)) * (1 / total_epochs) * 100
                self.session.commit()
    
            # Simulate a short sleep to mimic training time
    
        epoch_loss = running_loss / len(train_loader.dataset)
        job.add_loss(epoch_loss)  # Append loss to loss_history

        # Update progress
        job.progress = ((epoch + 1) / total_epochs) * 100
        self.session.commit()
        print(f"Job {job_id} progress: {job.progress}%")

    # Mark job as completed
    job.status = 'completed'
    job.stage_status = 'completed'
    job.progress = 100.0
    self.session.commit()
    print(f"Job {job_id} completed.")
    
@celery.task(name="jobapp.gpu_manager", ignore_result=True)
def gpu_manager():
    """
    Manager task that allocates queued jobs to available GPUs based on memory requirements.
    """
    def get_available_queues():
        i = celery.control.inspect()
        active_tasks = i.active()  # Dictionary: {worker: [tasks]}
        print("Active tasks: ", active_tasks)
        # Extract queues that are currently processing tasks
        busy_queues = set()
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    queue = task.get('delivery_info', {}).get('routing_key')
                    if queue:
                        busy_queues.add(queue)
        
        # All queues defined in configuration
        all_queues = set([q.name for q in celery.conf.task_queues])
        
        # Available queues are those not in busy_queues
        print("Busy queues: ", busy_queues)
        available_queues = all_queues - busy_queues
        return list(available_queues)
    available_queues = get_available_queues()
    print(f"Available queues: {available_queues}")
    # Fetch all queued jobs, sorted by creation time or job_id
    queued_jobs = Job.query.filter_by(status='queued').order_by(Job.created_at.asc()).all()
    if not queued_jobs:
        print("No jobs to process.")
        return  # No jobs to process
    
    # Get current GPU memory availability
    gpu_memory = get_gpu_memory()
    if not gpu_memory:
        print("No GPU information available.")
        return
    
    # available_queues = get_available_queues()
    if not available_queues:
        print("No available queues at the moment. Waiting for workers to free up.")
        return 
    else:
        print(f"Available queues: {available_queues}")

    for job in queued_jobs:
        memory_required = job.running_memory if job.running_memory else job.memory_required # using the running_memory if it is available, otherwise use the memory_required that is set by the user
        if memory_required == None:
            memory_required = 256 # some random amount of memory
        # Skip if already assigned
        if job.assigned_gpu is not None:
            continue

        # User has specified a GPU device
        if job.prefered_gpu is not None:
            # Find the GPU info
            gpu_info = next((gpu for gpu in gpu_memory if gpu['gpu_id'] == job.prefered_gpu), None) # get the info of prefered GPU
            if gpu_info and gpu_info['memory_free'] >= memory_required:
                # Assign GPU to job
                queue_prefix = f"gpu{job.prefered_gpu}"
                assigned_queue = None
                for possible_queue in available_queues:
                    if queue_prefix in possible_queue:
                        assigned_queue = possible_queue
                        break
                if assigned_queue is None:
                    print(f"GPU {job.prefered_gpu} does not have an available queue. for Job {job.job_id} with {memory_required}.")
                    continue
                        
                job.assigned_gpu = gpu_info['gpu_id']
                job.status = 'queued'  # Ensure status is 'queued'
                db.session.commit()

                # Dispatch the training task with assigned GPU
                visiontask.apply_async(args=[job.job_id, gpu_info['gpu_id']], queue=assigned_queue)
                available_queues.remove(assigned_queue) # remove the assigned queue from the available queues

                print(f"Assigned Job {job.job_id} to GPU {gpu_info['gpu_id']} to queue {assigned_queue}")
                continue
            else:
                print(f"GPU {job.assigned_gpu} does not have enough memory for Job {job.job_id} with {memory_required}.")
            continue

        # If GPU device is not specified, auto-assign based on memory
        for gpu in gpu_memory:
            if gpu['memory_free'] >= memory_required:
                # Assign GPU to job
                assigned_queue = None
                queue_prefix = f"gpu{gpu['gpu_id']}"
                for possible_queue in available_queues:
                    if queue_prefix in possible_queue:
                        assigned_queue = possible_queue
                        break
                if assigned_queue is None:
                    print(f"GPU {gpu['gpu_id']} does not have an available queue. for Job {job.job_id} with {memory_required}.")
                    continue
                
                job.assigned_gpu = gpu['gpu_id']
                job.status = 'queued'  # Ensure status is 'queued'
                db.session.commit()
                # Dispatch the training task with assigned GPU
                visiontask.apply_async(args=[job.job_id, gpu['gpu_id']], queue=assigned_queue)
                available_queues.remove(assigned_queue)

                print(f"Assigned Job {job.job_id} to GPU {gpu['gpu_id']} to queue {assigned_queue}")
                break  # Move to the next job after assignment
        else:
            # No GPU available for this job currently
            print(f"No available GPU for Job {job.job_id} at this time.")
            continue