# app/decorators.py

import torch
from functools import wraps
from jobapp.models import Job
from jobapp import db
import time
import threading
from flask import current_app as app

def record_peak_gpu_memory(func):
    # def decorator(func):
        @wraps(func)
        def wrapper(job_id, gpu_id, *args, **kwargs):
            # Reset peak memory stats before training
            job = Job.query.filter_by(job_id=job_id).first()
            print(f"Job {job_id}: Using GPU {gpu_id}")
            torch.cuda.reset_peak_memory_stats() # does not to specify device https://discuss.pytorch.org/t/how-to-calculate-the-gpu-memory-that-a-model-uses/157486/6
            # torch.cuda.empty_cache()
            # job.running_memory = 3440
            # db.session.commit() # check if this is necessary
            # with app.app_context():
            #     job.running_memory = 12312
            #     db.session.commit()
            # Function to monitor GPU memory
            def monitor_memory(app):
                peak_memory = 0
                while not stop_event.is_set():
                    print("Monitoring GPU memory")
                    current_memory = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 2)  # Convert to MB
                    if current_memory > peak_memory:
                        peak_memory = current_memory
                        # Update the database with the new peak memory
                        job.running_memory = int(peak_memory)
                        with app:
                            db.session.commit()
                        print(f"Job {job_id}: Updated peak GPU memory to {peak_memory:.2f} MB")
                    time.sleep(0.1)  # Adjust the interval as needed
                    # Event to signal the monitoring thread to stop
            
            stop_event = threading.Event()

            # Start the monitoring thread
            monitor_thread = threading.Thread(target=monitor_memory, args=[app.app_context()])
            monitor_thread.start()
            print("Job {job_id}: Starting training")
            error = None

            # Execute the training function
            try:
                result = func(job_id, gpu_id, *args, **kwargs)
            except Exception as e:
                # extract the stack trace
                import traceback
                print(traceback.format_exc())
                # Log the exception (optional: use logging instead of print)
                # print(f"Job {job_id} failed with error: {str(e)}")
                # clear all the memory
                # torch.cuda.empty_cache()

                # Update the Job model with failure reason and status
                # job = Job.query.filter_by(job_id=job_id).first()
                # if job:
                #     job.status = 'failed'
                #     job.stage_status = 'Error during executing'
                #     job.failure_reason = str(e)
                #     # print("Job {job_id}: Job failed")
                #     with app.app_context():
                #         db.session.commit()
                #     job = Job.query.filter_by(job_id=job_id).first()
                    # print(f"Job {job_id}: Job status: {job.status}, failure reason: {job.failure_reason}")
                result = {'message': 'Job failed', 'job_id': job_id}, 500
                error = e
                # stop_event.set()
                # monitor_thread.join()
                # return {'message': 'Job failed', 'job_id': job_id}, 500
            
            finally:
                # Update the Job model with failure reason and status
                if error is not None:
                    print("Error is not None")
                    torch.cuda.empty_cache()
                    job = Job.query.filter_by(job_id=job_id).first()
                    if job:
                        print("Job is not None, we enter here")
                        job.status = 'failed'
                        job.stage_status = 'Error during executing'
                        job.failure_reason = str(error)
                        print(f"Job {job_id}: Job failed")
                        with app.app_context():
                            db.session.commit()
                            db.session.close()
                        job = Job.query.filter_by(job_id=job_id).first()
                        print(f"Job {job_id}: Job status: {job.status}, failure reason: {job.failure_reason}")
                # Signal the monitoring thread to stop and wait for it to finish
                stop_event.set()
                monitor_thread.join()

            # Get peak memory usage in MB
            # peak_memory_bytes = torch.cuda.max_memory_allocated(gpu_id)
            # peak_memory_mb = peak_memory_bytes / (1024 ** 2)

            # # Update the Job model with running_memory
            # job = Job.query.filter_by(job_id=job_id).first()
            # if job:
            #     job.running_memory = int(peak_memory_mb)
            #     db.session.commit()
            #     print(f"Job {job_id}: Peak GPU Memory Usage: {peak_memory_mb:.2f} MB")

            return result
        return wrapper
    # return decorator 