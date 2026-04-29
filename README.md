# Image Classification with Deep Learning Using TensorFlow
**Mobility Vision – Traffic Sign Recognition Pipeline**

## Overview
The following details the end-to-end engineering of a deep learning computer vision pipeline designed to classify Traffic Sign Recognition images. This project is highly valuable for the mobility sector as the objective is to develop a robust Convolutional Neural Network (CNN) capable of identifying 43 distinct class of traffic signs. 50,000+ images were processed to achieve a 91% accuracy on unseen data.

Beyond baseline model accuracy, this serves as a comprehensive MLOps demonstration as it maps the transition from local exploratory data analysis to cloud accelerated GPU (A100) training. As the exploratory data analysis phase is decoupled from the cloud computer phase. A remote Dagshub server was established to track all hyperparameters, telemetry, and model artifacts in order to automatically log via MLflow.
The below is the complete lifecycle from raw pixel arrays to a deployment ready .h5 model artifact tracked and versioned using Mlflow and Dagshub.


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
├── .gitignore               
├── requirements.txt         
└── README.md
```

## Getting Started

### Dependencies:
* Python 3.13.5
* TensorFlow
* Scikit-Learn
* Pandas
* NumPy
* Keras
* Current Web Browser
* Jupyter Notebook
* Google Colab Pro
* Google Drive
* DagsHub

### Executing Program
1. Configure MLflow & DagsHub
    * This will allow for experiment racking and the ability to save the model as an artifact. When running in Google Colab, store your DagsHub Default Access Token in Colab’s native Secrets manager to secure token data
2. Acquire Data & Preprocessing
    * Download the Traffic Sign dataset and compress it into the file name TRAFFIC_SIGN.zip
    * Running on Google Colab, upload this zip file to the root of your Google Drive
    * Initial notebook cells in Colab will mount your Google Drive, extract the contents, and verify the file structure.
3. Model Training (Colab)
    * Execute the Colab cells sequentially, the pipeline will handle several critical data engineering tasks:
4. Evaluation & Error Analysis
    * Run the validation/test cells in order to evaluated the model on unseen data.

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



## TODO

While the current iteration of the pipeline achieves a viable 91% real-world accuracy. The following enhancements are scheduled for future development phases to harden the model for true edge deployment:

- Data Augmentation for Minority Classes: Implement real-time Keras image augmentation (targeted rotation, zoom, shear, and brightness shifts) specifically focused on low-support classes. This will synthetically expand the dataset and improve recall on rare traffic signs.
- Edge Optimization (TF Lite): Quantize the final .h5 model weights and convert the artifact to TensorFlow Lite. This will drastically reduce the memory footprint and inference latency, enabling the model to run on embedded vehicle hardware rather than requiring cloud compute.
- Automated Hyperparameter Tuning: Integrate KerasTuner to systematically sweep and optimize the CNN architectural parameters (filter sizes, optimal dropout rates, dense layer neurons) to push the baseline accuracy closer to 99%.
- FastAPI / Streamlit Interface: Develop a lightweight web application or API endpoint to serve the model, allowing users to upload raw dashcam images and receive real-time class predictions alongside confidence probability scores.



## Authors
* Lead Developer – **Eric Serrano**

## Version History
* 0.1 – Initial Release 

## License
The MIT License (MIT)
Copyright (c) 2026 Eric Serrano

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.