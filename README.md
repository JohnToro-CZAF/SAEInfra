![image](https://github.com/user-attachments/assets/b9d0a1cc-87cb-4c50-8c6b-04d3d184f439)

# Dynamicaly allocating inference or training torch instance on GPU cluster
## Built with Flask, Celery and Redis
Anyone has problem of running multiple deep learning experiments on highly available GPU cluster. Scheduling the jobs on basis

* Allows to stop, continue the job instance.
* Dynamically calculating required GPU memory, and retry the job if it is failed.
* Multiple jobs on multiple GPUs
* Queue-based job executing
* Allows threading-based job instance
* Each Celery worker (Thread POOL implementation) handles a job at one time. Since this is good for large-scaled, long running time like inference or training instance. 
TODO:
* Allows to configure priority.
* Allows to configure the running task:
    * Save method: saving when the job is stopped.
    * Load method: loading the job context if it is continued.
    * Log method: to logs to database the necessary information about the job.
    * Run method: running any black box torch-based function.
* Dynamically scaling workers
    * Low demand: kill idle workers on different GPUs  
    * High demand: launching multiple workers with preset maximum capacity. Others job instances must be queued.
``` bash
redis-server
sh startworkers.sh
```

``` python
python manage.py
```
