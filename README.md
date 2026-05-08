# Image Classification with Deep Learning Using TensorFlow
**Mobility Vision – Traffic Sign Recognition Pipeline**

**Traffic Vision API - Serverless Image Classification**

## Project Overview
This project demonstrates the complete engineering lifecycle of a deep learning computer vision pipeline designed for the Mobility and Autonomous Transportation sector. The core objective is the development and deployment of a robust Convolutional Neural Network (CNN) capable of identifying 43 distinct classes of traffic signs with high precision. By processing a dataset of over 50,000 images, the model achieved a 91% accuracy rate on unseen validation data, providing a reliable foundation for real-world traffic sign recognition.
Engineering & MLOps Lifecycle

Beyond baseline predictive accuracy, this repository serves as a comprehensive MLOps demonstration, mapping the transition from local exploratory data analysis (EDA) to high-performance cloud computing:

- **Accelerated Training:** The training phase is decoupled from local hardware, utilizing Cloud GPU (A100) acceleration to handle the high-dimensional pixel arrays and complex weight optimizations of the CNN.

- **Experiment Tracking & Governance:** A remote DagsHub server was established to maintain a rigorous record of the project’s evolution. Utilizing MLflow, all hyperparameters, training telemetry, and model artifacts were automatically logged, versioned, and tracked, ensuring full reproducibility.

- **Production-Grade Serving:** The lifecycle culminates in a production-ready serverless REST API built with FastAPI. This service allows external systems to submit image payloads and receive categorical predictions with associated confidence scores in real-time.

**Deployment & Scalability**

Designed with modern infrastructure standards, the application is fully containerized via Docker and deployed on Google Cloud Run. This architecture bridges the gap between a static .h5 model artifact and a globally accessible, horizontally scalable cloud endpoint. The result is a highly portable and resilient system tailored specifically for the demands of modern mobility and transportation data pipelines.


## Repository Structure
```text
image-classification-mobility-vision/
│
├── data/
│   ├── Meta/          
│   └── Test/
│   └── Train/             
│   └── Meta.csv
│   └── Test.csv
│   └── Train.csv
│
├── notebooks/
│   ├── 01_EDA_Testing.ipynb       
│   └── 02_Colab_Env.ipynb          
│
├── deployment/     
│   ├── models/                
│   │   └── latest_checkpoint.h5  
│   │
│   ├── app.py     
│   ├── Dockerfile        
│   ├── requirements.txt 
│   └── .gcloudignore 
│
├── tests/
│   └── test_api.py
│
├── .gitignore               
└── README.md
```

## Getting Started

### Dependencies:
This project utilizes a branched technology stack to handle both high-compute deep learning training and lightweight, serverless inference.

Core Languages & Data Science

* Python 3.13.5
* TensorFlow 
* Keras
* Scikit-Learn
* Pandas
* NumPy
* Pillow (PIL)

MLOps & Experiment Tracking

* DagsHub
* MLflow
* Google Colab Pro
* Jupyter Notebook

Production Infrastructure (The Deployment Stack)

* FastAPI
* Uvicorn
* Docker
* Google Cloud Run
* Google Cloud Build
* Artifact Registry

### Executing Program
The project is executed in three distinct phases, moving from exploratory data science to scalable cloud production.

**Phase 1: Research & Model Training**

1. Environment: Open the notebooks in notebooks/ via Google Colab Pro or a local Jupyter instance.

2. Experiment Tracking: Ensure the DagsHub/MLflow URI is configured to track telemetry.

3. Execution: Run the training pipeline to process the 50,000+ image dataset using A100 GPU acceleration.

4. Artifact Generation: The finalized .h5 model weights are versioned in MLflow and exported to the deployment/models/ directory for the next phase.

**Phase 2: Containerization & Cloud Deployment**

This phase transforms the static model into a dynamic service.

1. Build Context: Navigate to the deployment/ directory.

2. Containerize: Build the Docker image locally or via Google Cloud Build:
    
    *Using Bash*

    - gcloud builds submit --tag gcr.io/[PROJECT_ID]/traffic-vision-api .

3. Deploy: Push the container to Google Cloud Run, ensuring the --memory 2Gi flag is used to accommodate the TensorFlow footprint.

4. Verification: Confirm the service is active via the GCP Console and monitor the initial "cold start" logs.

**Phase 3: Production Inference**

Once deployed, the model can be consumed through two primary interfaces:

1. Interactive Testing: Navigate to the /docs endpoint of your Cloud Run URL to use the FastAPI Swagger UI for manual image uploads.

2. Programmatic Testing: Execute the test_api.py script from a local terminal to simulate a real-world machine-to-machine request:
    
    *Using Bash*

    - python tests/test_api.py

## Expected Results

As part of the exploratory data analysis (EDA), an integrity check was conducted in order to cross reference the file counts against the rows in the CSVs. This allows for an understanding of whether there is any data missing that would crash the pipeline training. 

![Integrity Check](/images/img1.png)

Continuing on with the EDA, the distribution of classes was visualized to understand if the distribution of data is heavily skewed. As the chart below highlights, there is a class imbalance which has been taken into account for when the architecture was developed. In the context of mobility, this check is particularly critical as a low frequency of a particular traffic sign could lead to dangerous real-world situations. 

![Distribution](/images/img2.png)

The final step of the EDA is to visualize a sample from the first 5 classes. The path for the first image in each class was fetched, the PNG images were than loaded and display as a visual verification. 

![Sample Check](/images/img3.png)

With the exploratory data analysis complete, the model testing phase begins. A single overfit batch test was intentionally conducted to ensure the network architecture was sound and all was configured correctly. A very simple version of the model was tested before the full 50,000 image training loop took place.

As the output below shows, the loss plummeted after the model was fine tuned. The final Epoch of the 25 Epoch run has an 80% accuracy rate which validates the next step of moving from testing into the training environment. 

![Epochs](/images/img4.png)

The following images now show the automated tracking as set up using DagHub and MLflow to provide a centralized, hosted environment for tracking the experiments. The model pipeline was developed using Google Colab environment. 
Two runs will be visible in the logs with the metrics to follow for the second, more successful run. 

**98% Validation Accuracy / 10 Epoch**


![Model Log](/images/img5.png)

![Model Results 1](/images/img6.png)

![Model Results 2](/images/img7.png)

![Model Metrics](/images/img8.png)

## Deployment

The inference endpoint is currently live and can be consumed via any standard HTTP client. 

1. Local Requirements

    - To run the local test script, you only need the requests library installed. The heavy machine learning dependencies are isolate within the cloud container. 
        - pip install requests

2. Execution Programmatic 

    - Ensure the test_api.py scription contains the live Cloud Run URL and execute it via a terminal. 
        - Python test_api.py 

3. Expected Response

    - The serverless container will process the image tensor and return a structured JSON response indicating the classification and confidence metric:

![console output](/images/img9.png)

4. Execution Interactive UI (Swagger)
    - Navigate to provided Cloud Run URL
    - Expand the POST/predict route.
    - Click Try it out and upload a raw image file of a traffic sign and press Execute. 

![UI overview](/images/img10.png)

5.	Observability & Monitoring

    - Beyond standard deployment, this API utilizes Google Cloud Run’s native observability suite to monitor system health and model inference performance in real time. 
        - Live Telemetry: The built in Metrics dashboard tracks active HTTP requests providing visual confirmation of request volume and inference latencies.
        - Serverless Auto Scaling: The infrastructure is configured for dynamic traffic management automatically scaling from 0 up to 20 container instances based on demand ensuring high availability while minimizing idle compute costs. 
        - Continuous Health Tracking: Native integration with Google Cloud Logging captures container lifecycle events and Python stack traces ensuring the Uvicorn ASGI server maintains a healthy, active state.

![GCP Metrics](/images/img11.png)


## Help / Issue Log

1. Validation Leak

    - Symptom: During the initial full-scale training run, training accuracy reached above 90% but validation accuracy flatlined at around 17% with a high validation loss greater than 39.0

    - Root Cause: The Train.csv file is sequentially sorted by ClassId and by default, the Keras validation split of 0.2 argument grabs the bottom 20% of the DataFrame. This resulted in a training set of classes 0-34 and a validation set of completely unseen classes 35-42, making it impossible for the model to validate correctly.

    - Resolution: Implemented a Pandas shuffle prior to generator ingestion. This successfully distributing all 43 classes evenly across both splits and raised validation accuracy to around 98%.

2. Gradient Descent on Initial Weights

    - Symptom: When running local architecture tests, the loss hovered at around 3.76 with 0% accuracy.

    - Root Cause: The initial optimizer configuration utilized a very small learning rate (1e-5) based on trial and error from a previous transfer-learning project. While ideal for fine-tuning pre-trained models like RetinaNet, it was too restrictive for a CNN initializing from scratch with random weights.

    - Resolution: Calibrated the starting learning rate of the CosineDecayRestarts scheduler to the industry standard 1e-3 for from scratch training. This allowed the Adam optimizer to properly navigate the loss gradients.

3. Keras Generator Shuffling During Local Testing

    - Symptom: Attempts to intentionally overfit a single batch of 5 images prior to moving to the Cloud Environment, failed to memorize the batch.

    - Root Cause: Calling steps per epoch as 1 on an ImageDataGenerator with ‘shuffle=True’ resulted in the model receiving a completely different batch of 5 images every epoch preventing memorization.

    - Resolution: Bypassed the generator for the pure architecture test by extracting a static array ‘(x_batch, y_batch = next(train_generator))’ and passing it directly to ‘model.fit()’.

4. Cloud Extraction Pathing (FileNotFoundError)

    - Symptom: Transitioning from local Jupyter testing to Colab resulted in pathing errors when Pandas attempted to read the CSV files.

    - Root Cause: Discrepancies between local Windows .zip extraction behavior and Colab's native zipfile extraction behavior.

    - Resolution: Developed a dynamic pathing check using ‘os.listdir('/content/data/')’ to verify the directory structure prior to instantiating the DataFrames, ensuring robust execution regardless of the cloud environment's specific unpacking protocols.

5. Build Pipeline: IAM Permission Denied

    - Symptom: Cloud Build failed immediately with a PERMISSION_DENIED error when attempting to fetch the source code from the staging bucket.

    - Root Cause: By default new Google Cloud projects provision service accounts with zero inherent permissions as a security measure. The default compute robot could not read the uploaded ZIP payload.

    - Resolution: Utilized the gcloud CLI to manually bind the roles/storage.admin and roles/artifactregistry.writer IAM policies to the default compute service account, authorizing the build steps.

6. Infrastructure: Port 8080 Timeout

    - Symptom: The container built successfully but failed deployment with a generic "Failed to start and listen on PORT=8080" error.

    - Root Cause: Cloud Run defaults to a highly restrictive memory allocation (512MB). Loading the heavy TensorFlow/Keras .h5 model weights caused a silent Out of Memory crash before the Uvicorn server could even open the port.

    - Resolution: Executed a vertical scaling operation via the deployment flag --memory 2Gi, providing sufficient RAM to cache the model in memory.

7. Framework: ASGI Module Import Error

    - Symptom: Cloud Run logs indicated: Error loading ASGI app. Could not import module "app". Container exited with status 1.

    - Root Cause: The main FastAPI script was locally named api.py. However, the Dockerfile command (CMD ["uvicorn", "app:app"]) specifically instructed the ASGI server to look for an entry point named app.py.

    - Resolution: Standardized the local and containerized file by renaming api.py to app.py, ensuring environmental parity.



## TODO

While the current iteration of the pipeline achieves a viable 91% real-world accuracy. The following enhancements are scheduled for future development phases to harden the model for true edge deployment:

- Data Augmentation for Minority Classes: 
    - Implement real-time Keras image augmentation (targeted rotation, zoom, shear, and brightness shifts) specifically focused on low-support classes. This will synthetically expand the dataset and improve recall on rare traffic signs.

- Edge Optimization (TF Lite): 
    - Quantize the final .h5 model weights and convert the artifact to TensorFlow Lite. This will drastically reduce the memory footprint and inference latency, enabling the model to run on embedded vehicle hardware rather than requiring cloud compute.

- Automated Hyperparameter Tuning: 
    - Integrate KerasTuner to systematically sweep and optimize the CNN architectural parameters (filter sizes, optimal dropout rates, dense layer neurons) to push the baseline accuracy closer to 99%.

- FastAPI / Streamlit Interface: **(Complete)** 
    - Develop a lightweight web application or API endpoint to serve the model, allowing users to upload raw dashcam images and receive real-time class predictions alongside confidence probability scores.

- Repository Consolidation: **(Complete)** 
    - Merge this deployment architecture with the original exploratory data science codebase to create a single unified repo for the complete mobility practice lifecycle.

- Dictionary Synchronization: 
    - Refactor the API's class_names mapping to strictly align with the class_indices generated during local model compilation resolving the minor categorical mapping drift identified in production.

- CI/CD Pipeline Integration: 
    - Implement GitHub Actions to automate the Google Cloud Build and Cloud Run deployment processes, enabling continuous delivery whenever changes are pushed to the main branch.

- Batch Inference Endpoint: 
    - Develop a secondary /predict_batch route within FastAPI to allow downstream systems to submit multiple images simultaneously for high-throughput processing.


## Authors
* Lead Developer – **Eric Serrano**

## Version History
* 1.1.0 (Current) – Monorepo Consolidation

    * Unified model research and deployment infrastructure into a single repository.

    * Standardized directory structure for MLOps best practices.

    * Updated comprehensive documentation and execution workflows.

* 1.0.0 – Production Release

    * Containerized the inference engine using Docker.

    * Deployed serverless REST API to Google Cloud Run with 2Gi RAM scaling.

    * Implemented FastAPI Swagger UI for interactive testing.

* 0.1.0 – Initial Research & Training

    * Developed Convolutional Neural Network (CNN) architecture for 43-class recognition.

    * Executed A100-accelerated training on 50,000+ images (91% accuracy).

    * Integrated MLflow and DagsHub for experiment tracking and artifact versioning.

## License
The MIT License (MIT)
Copyright (c) 2026 Eric Serrano

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
