// app.js

document.addEventListener('DOMContentLoaded', () => {
    const createJobForm = document.getElementById('create-job-form');
    const jobsContainer = document.getElementById('jobs-container');

    // Initialize Socket.IO client
    // const socket = io('http://localhost:5050');

    // Fetch and display all jobs on load
    fetchJobs();

    // Listen for form submission to create a new job
    createJobForm.addEventListener('submit', (e) => {
        console.log('Form submitted');
        e.preventDefault();
        const modelName = document.getElementById('model-name').value.trim();
        const datasetName = document.getElementById('dataset-name').value.trim();

        if (modelName && datasetName) {
            createJob(modelName, datasetName);
        }
    });

    // Handle real-time progress updates
    
    // socket.on('progress', (data) => {
    //     const { job_id, progress } = data;
    //     console.log('Progress update on job_id:', data);
    //     updateJobProgress(job_id, progress);
    // });

    // Function to fetch all jobs from the backend
    function fetchJobs() {
        fetch('http://localhost:5000/api/jobs')
            .then(response => response.json())
            .then(data => {
                const jobs = data.jobs;
                jobsContainer.innerHTML = ''; // Clear existing jobs
                jobs.forEach(job => {
                    addJobCard(job);
                });
            })
            .catch(error => console.error('Error fetching jobs:', error));
    }

    // Function to create a new job
    function createJob(modelName, datasetName) {
        console.log('Creating job:', modelName, datasetName);
        fetch('http://localhost:5000/api/jobs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: modelName,
                dataset_name: datasetName
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Job created:', data);
            fetchJobs(); // Refresh job list
            createJobForm.reset(); // Reset the form
        })
        .catch(error => console.error('Error creating job:', error));
    }

    // Function to add a job card to the DOM
    function addJobCard(job) {
        const card = document.createElement('div');
        card.className = 'card job-card';
        card.id = `job-${job.job_id}`;

        card.innerHTML = `
            <div class="card-body">
                <h5 class="card-title">Job ID: ${job.job_id}</h5>
                <p class="card-text"><strong>Model:</strong> ${job.model_name}</p>
                <p class="card-text"><strong>Dataset:</strong> ${job.dataset_name}</p>
                <p class="card-text"><strong>Status:</strong> <span id="status-${job.job_id}">${job.status}</span></p>
                <div class="progress mb-3">
                    <div class="progress-bar" role="progressbar" id="progress-${job.job_id}" style="width: ${job.progress}%;" aria-valuenow="${job.progress}" aria-valuemin="0" aria-valuemax="100">${job.progress}%</div>
                </div>
                <div id="controls-${job.job_id}">
                    ${job.status === 'running' ? `
                        <button class="btn btn-danger btn-sm" onclick="stopJob('${job.job_id}')">Stop</button>
                    ` : job.status === 'stopped' ? `
                        <button class="btn btn-success btn-sm" onclick="continueJob('${job.job_id}')">Continue</button>
                    ` : job.status === 'completed' ? `
                        <button class="btn btn-secondary btn-sm" disabled>Completed</button>
                    ` : job.status === 'queued' ? `
                        <button class="btn btn-secondary btn-sm" disabled>Queued</button>
                    ` : ''
                    }
                </div>
            </div>
        `;

        jobsContainer.appendChild(card);
    }

    // Function to update job progress and status
    function updateJobProgress(jobId, progress) {
        const progressBar = document.getElementById(`progress-${jobId}`);
        const statusSpan = document.getElementById(`status-${jobId}`);
        const controlsDiv = document.getElementById(`controls-${jobId}`);

        if (progressBar && statusSpan) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
            progressBar.textContent = `${progress.toFixed(2)}%`;

            // Update status based on progress
            if (progress >= 100) {
                statusSpan.textContent = 'completed';
                controlsDiv.innerHTML = ''; // Remove controls
            } else if (progress > 0 && progress < 100) {
                statusSpan.textContent = 'running';
                controlsDiv.innerHTML = `
                    <button class="btn btn-danger btn-sm" onclick="stopJob('${jobId}')">Stop</button>
                `;
            }
        }
    }

    // Expose stopJob and continueJob to the global scope
    window.stopJob = function(jobId) {
        fetch(`http://localhost:5000/api/jobs/${jobId}/stop`, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Job stopped:', data);
            fetchJobs(); // Refresh job list
        })
        .catch(error => console.error('Error stopping job:', error));
    };

    window.continueJob = function(jobId) {
        fetch(`http://localhost:5000/api/jobs/${jobId}/continue`, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Job continued:', data);
            fetchJobs(); // Refresh job list
        })
        .catch(error => console.error('Error continuing job:', error));
    };
});
