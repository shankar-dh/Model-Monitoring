## Model Monitoring Lab

This project outlines the steps for setting up a Machine Learning Operations (MLOps) pipeline using Google Cloud services for boston housing dataset and then creating a model monitoring dashboard using Looker studio.

## Project Configuration
**⚠️ Important** <br>
Goto IAM & Admin > service accounts and create a service account with the following roles:
- Storage Admin
- Vertex AI Admin
- AI Platform Admin
- Artifact Registry Administrator
- Service Account Admin
- BigQuery Admin
- Logging Admin

Download the json file for the service account and save it in your system safely. You will need this file to authenticate the service account. To authenticate the service account run the following command:
```bash
gcloud auth login
gcloud auth application-default login
gcloud auth activate-service-account --key-file=service_account.json
```
Replace `service_account.json` with the name of the json file you downloaded.

Create a `.env` file in the root of your project directory with the following content. This file should not be committed to your version control system so add it to your `.gitignore` file. This file will be used to store the environment variables used in the project. You can change the values of the variables as per your requirements.
```
# Google Cloud Storage bucket name
BUCKET_NAME= [YOUR_BUCKET]

# Google Cloud Storage bucket directory for storing the data
BASE_OUTPUT_DIR=gs://[YOUR_BUCKET]

# Google Cloud AI Platform model directory
AIP_MODEL_DIR=gs://[YOUR_BUCKET]/model

# Google Cloud region
REGION=us-east1

# Google Cloud project ID
PROJECT_ID=[YOUR_PROJECT_ID]

# Container URI for training
CONTAINER_URI=us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/trainer:v1

# Container URI for model serving
MODEL_SERVING_CONTAINER_IMAGE_URI=us-east1-docker.pkg.dev/YOUR_PROJECT_ID/[FOLDER_NAME]/serve:v1

# Health check route for the AI Platform model
AIP_HEALTH_ROUTE=/ping

# Prediction route for the AI Platform model
AIP_PREDICT_ROUTE=/predict

# Service account key file path
SERVICE_ACCOUNT_EMAIL= [YOUR_SERVICE_ACCOUNT_EMAIL]

# Bigquery path
BIGQUERY_PATH= [YOUR_BIGQUERY_PATH] # example: [YOUR_PROJECT_ID].[DATASET].[TABLE]
```
You will get to know about the usage of these variables in the later sections of the project. [YOUR_PROJECT_ID] should be the name of your GCP project ID, [FOLDER_NAME] should be the name of the folder in which you want to store the docker images in the Artifact Registry. [YOUR_SERVICE_ACCOUNT_EMAIL] should be the email address of the service account you created.

Ensure you have all the necessary Python packages installed to run the project by using the `requirements.txt` file. Run the following command to install the packages.
```bash
pip install -r requirements.txt
```

## Data Source
The dataset used for this project is the [Boston Housing Dataset](https://www.kaggle.com/c/boston-housing). Please download the dataset from the link and save it in the data folder in your GCS bucket.

## Data Preprocessing
The `data_preprocess.py` script automates dataset preparation for machine learning workflows, with a specific emphasis on MEDV as the target variable. Upon execution, this script generates training and testing datasets and subsequently stores them in your Google Cloud Storage (GCS) bucket.

**Normalization**  
Normalization statistics are computed from the training data and stored in GCS at `gs://YOUR_BUCKET/scaler/normalization_stats.json`. These statistics are updated in tandem with the datasets to reflect the current scale of the training data, providing consistency in the data fed into the model.


## Model Training, Serving and Building
1. **Folder Structure**:
    - The main directory is `src`.
    - Inside `src`, there are two main folders: `trainer` and `serve`.
    - Each of these folders has their respective Dockerfiles, and they contain the `train.py` and `predict.py` code respectively.
    - There's also a `build.py` script in the main directory which uses the `aiplatform` library to build and deploy the model to the Vertex AI Platform and a `inference.py` script which can be used to generate predictions from the model.
```
src/
|-- trainer/
|   |-- Dockerfile
|   |-- train.py
|-- serve/
|   |-- Dockerfile
|   |-- predict.py
|-- build.py
|--inference.py
```

### Model Training Pipeline (`trainer/train.py`)
This script orchestrates the model training pipeline, which includes data retrieval, preprocessing, model training, and exporting the trained model to Google Cloud Storage (GCS).

#### Detailed Workflow
- **Data Retrieval**: It starts by fetching the dataset from a GCS bucket using the `load_data()` function. The dataset is then properly formatted with the correct column headers.
- **Preprocessing**: The `data_transform()` function is responsible for converting date and time into a datetime index, splitting the dataset into training and validation sets, and normalizing the data using the pre-computed statistics stored in GCS We also calculate the min and max values for each of the features and store them as a json file in GCS. This json file will be used to check for anomalies in the prediction data.
- **Model Training**: A RandomForestRegressor model is trained on the preprocessed data with the `train_model()` function, which is designed to work with the features and target values separated into `X_train` and `y_train`.
- **Model Exporting**: Upon successful training, the model is serialized to a `.pkl` file using `joblib` and uploaded back to GCS. The script ensures that each model version is uniquely identified by appending a timestamp to the model's filename.
- **Normalization Statistics**: The normalization parameters used during preprocessing are retrieved from a JSON file in GCS, ensuring that the model applies consistent scaling to any input data.
- **Version Control**: The script uses the current time in the US/Eastern timezone to create a version identifier for the model, ensuring traceability and organization of model versions.

#### Notes on Environment Configuration:
- A `.env` file is expected to contain environment variables `BUCKET_NAME` and `AIP_MODEL_DIR`, which are crucial for pointing to the correct GCS paths.
- The Dockerfile within the `trainer` directory specifies the necessary Python environment for training, including all the dependencies to be installed.

#### Docker Environment Configuration
The Dockerfile located within the `trainer` directory defines the containerized environment where the training script is executed. Here's a breakdown of its content:
- **Base Image**: Built from `python:3.9-slim`, providing a minimal Python 3.9 environment.
- **Working Directory**: Set to the root (`/`) for simplicity and easy navigation within the container.
- **Environment Variable**: `AIP_STORAGE_URI` is set with the GCS path where the model will be stored and 'BUCKET_NAME' is set with the GCS bucket name.
- **Copying Training Script and service account**: The training script located in the `trainer` directory is copied into the container and the service account json file is copied into the container. Without this service account the training job will not be able to write logs.
- **Installing Dependencies**: Only the necessary dependencies that are relevant to the training code are installed to minimize the Docker image size.
- **Entry Point**: The container's entry point is set to run the training script, making the container executable as a stand-alone training job.

The provided Docker image is purpose-built for training machine learning models on Google Cloud's [Vertex AI Platform](https://cloud.google.com/vertex-ai?hl=en). Vertex AI is an end-to-end platform that facilitates the development and deployment of ML models. It streamlines the ML workflow, from data analysis to model training and deployment, leveraging Google Cloud's scalable infrastructure.

Upon pushing this Docker image to the Google Container Registry (GCR), it becomes accessible for Vertex AI to execute the model training at scale. The image contains the necessary environment setup, including all the dependencies and the `trainer/train.py` script, ensuring that the model training is consistent and reproducible. 

### Model Serving Pipeline (`serve/predict.py`)
This script handles the model serving pipeline, which includes loading the most recent model from Google Cloud Storage (GCS), preprocessing incoming prediction requests, and providing prediction results via a Flask API. Before running the script make sure you have created a new dataset in your Bigquery table under the same project.

#### Detailed Workflow
- **Model Loading**: 
  At application startup, the `fetch_latest_model` function is tasked with retrieving the latest model file from the designated GCS bucket. This is determined by identifying the model file with the most recent timestamp.
- **Normalization Statistics**: 
  The system loads normalization statistics from a JSON file in GCS. These statistics are critical for consistently normalizing the prediction input data, ensuring data fed into the model is in the correct format.
- **Flask API**: 
  The Flask application offers two key endpoints:
  - `/ping`: A health check endpoint to verify the application's operational status.
  - `/predict`: This endpoint handles POST requests with input instances for predictions. It logs all prediction data to BigQuery for monitoring and analysis. Additionally, it records any instances where prediction data falls outside the min and max value range for each feature as a warning. This BigQuery table will be utilized to create a monitoring dashboard in Looker Studio.
- **Data Preprocessing**: 
  Incoming prediction request data is normalized using the `normalize_data` function. This function applies the normalization statistics previously loaded to ensure data consistency.
- **Prediction**: 
  The RandomForestRegressor model processes the normalized data to generate predictions. These predictions are then formatted and returned as a JSON response.


#### Environment Configuration Notes:
- The environment variables `AIP_STORAGE_URI`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE`, and `AIP_HTTP_PORT` are set within the Dockerfile and are crucial for defining the GCS path, API routes, and the port the Flask app runs on.
- The Docker environment configuration specifies the necessary Python environment for the Flask application, including all dependencies that are installed.

#### Docker Environment Configuration
- **Base Image**: Utilizes `python:3.9-slim` as a minimal Python 3.9 base image.
- **Working Directory**: The working directory in the container is set to `/app` for a clean workspace.
- **Environment Variables**: Sets up environment variables needed for the Flask app to correctly interface with Google Cloud services and define API behavior.
- **Copying Serving Script and service account**: The `predict.py` script is copied from the local `serving` directory into the container's `/app` directory. Along with this, the service account json file is also copied into the container. Without this service account the serving job will not be able to write logs as well as access the Bigquery table.
- **Installing Dependencies**: Only the necessary dependencies that are relevant to the serving code are installed to minimize the Docker image size.
- **Entry Point**: The Dockerfile's entry point is configured to run the `predict.py` script, which starts the Flask application when the container is run.

### Pushing Docker Images to Google Artifact Registry
Download Google cloud SDK based on your OS from [here](https://cloud.google.com/sdk/docs/install) and ensure Docker daemon is running. Follow the below steps to push the docker images to Google Artifact Registry.

- **Step 1: Enable APIs**
    1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
    2. Navigate to **APIs & Services** > **Library**.
    3. Search for and enable the **Artifact Registry API** and **Vertex AI API**.

- **Step 2: Set Up Google Cloud CLI**
    1. Install and initialize the Google Cloud CLI to authenticate and interact with Google Cloud services from the command line.
        ```bash
        gcloud auth login
        gcloud config set project [YOUR_PROJECT_ID]
        ```

- **Step 3: Configure Docker for GCR**
    1. Configure Docker to use the gcloud command-line tool as a credential helper:
        ```bash
        gcloud auth configure-docker us-east1-docker.pkg.dev
        ```

- **Step 4: Create a GCR Account and Repository Folders**
    1. In the Google Cloud Console, open the Artifact Registry.
    2. Create repositories using a naming convention for `[FOLDER_NAME]`.

- **Step 5: Build and Push the Training Image**
    1. Navigate to the src directory and run:

    ```bash
    docker buildx build --platform linux/amd64 -f trainer/Dockerfile -t us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/trainer:v1 . --load
    docker push us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/trainer:v1
    ```

- **Step 6: Build and Push the Serving Image**
    1. Navigate to the src directory and run:

    ```bash
    docker buildx build --platform linux/amd64 -f serve/Dockerfile -t us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/serve:v1 . --load
    docker push us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/serve:v1
    ```

    **Note:** 
    > Run the above docker bash codes inside the src directory. <br>

**Reference:** <br>
[Create a custom container image for training (Vertex AI)](https://cloud.google.com/vertex-ai/docs/training/create-custom-container)

### Building and Deploying the Model (`build.py`)
The `build.py` script is responsible for building and deploying the model to the Vertex AI Platform. It uses the `aiplatform` library to create a custom container training job and deploy the model to an endpoint.  The CustomContainerTrainingJob class is a part of Google Cloud's Vertex AI Python client library, which allows users to create and manage custom container training jobs for machine learning models. A custom container training job enables you to run your training application in a Docker container that you can customize.

#### Steps to Build and Deploy the Model
1. **Initialize AI Platform**: With the `initialize_aiplatform` function, the Google Cloud AI platform is initialized using the project ID, region, and staging bucket details. 

2. **Create Training Job**: The `create_training_job` function creates a custom container training job using the `CustomContainerTrainingJob` class.

3. **Run Training Job**: The `run_training_job` function runs the training job using the `run` method of the `CustomContainerTrainingJob` class. This also binds the service account to the AI Platform which is crucial for the training job to access the GCS bucket and other Google Cloud services.

4. **Deploy Model**: After the training job completes, the `deploy_model` function deploys the trained model to an AI Platform endpoint, using the provided model display name. 

- When you run this code the aiplatform library will create a custom container training job and deploy the model to an endpoint. It uses both the dockerfiles to build the training and serving images. The training image is used to train the model and the serving image is used to serve the model.

> There are multiple configuration options available in vertex AI other than these parameters. Please refer to the [documentation](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomContainerTrainingJob) for more information. <br>

The advantages of using a custom training job over a normal training job in Vertex AI include:
- Greater flexibility with container images, allowing for specific dependencies and configurations.
- More control over the training environment and resources.
- Ability to integrate with existing CI/CD workflows and toolchains.
- Custom training logic and advanced machine learning experimentation that might not be supported by standard training options.

Once the model is deployed to an endpoint, you can utilize the `inference.py` script to generate predictions from the model. The `predict_custom_trained_model` function serves as a REST API endpoint, enabling you to obtain predictions by sending it instance data. This function constructs a prediction request tailored to the model's expected input schema and transmits it to the deployed model on Vertex AI. The resulting prediction outcomes are then displayed.

**Reference:** <br>
[Custom Service Account Configuration](https://cloud.google.com/vertex-ai/docs/general/custom-service-account) <br>
[Deploying Model to endpoint](https://cloud.google.com/vertex-ai/docs/general/deployment) <br>
[Online Predictions from Custom Model](https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions) <br>
[Batch Prediction from Custom Trained Model](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions) <br>
[Custom Container Training Job](https://cloud.google.com/vertex-ai/docs/training/create-custom-container) <br>



## Model Monitoring
Create a new account in [Looker](https://looker.com/) and create a new project. Go to Data Sources -> Create -> Data Source -> Bigquery and connect your Bigquery table to Looker. Select the table you created in the Bigquery dataset. Set the Data freshness to 1 minute or 1 hour as per your requirements and click on creaate report. 

### Creating a Dashboard
- **Feature Value vs. Timestamp Plot**:
  - **Setup**: Plot each feature against the timestamp on the x-axis and the feature value on the y-axis. Select the timestamp and desired feature for plotting, and choose 'table' as the chart type.
  - **Threshold Lines**: Utilize the min and max values saved in the JSON file to create threshold lines for each feature. For instance, if the minimum value for a feature is 0 and the maximum is 100, create threshold lines at these points.
  - **Conditional Formatting**: Navigate to `Style -> Conditional Formatting -> Add` in the table options. Select the relevant column and use the JSON file to set conditions: 'greater than' the max value and 'less than' the min value from your training data. Choose red for the color indicator.
- **Timestamp vs. Features with Min/Max Values (Line Chart)**:
  - **Configuration**: In the Date Range dimension, select 'timestamp', and under Metric, choose the feature and sort in ascending order.
  - **Reference Lines**: In the style tab, add reference lines corresponding to the min and max values for that feature, as determined by your training data.
- **Total Number of Predictions Plot**:
  - **Visualization**: Display the total number of predictions for the endpoint as a table by selecting the 'Record count'.
- **Average Latency Plot**:
  - **Setup**: Choose 'latency' to plot and display it as a table.
  - **Aggregation**: In the Metric tab, select the average (avg) for latency.
- **Additional Plot Suggestions**:
  - **Total Number of Errors**: Create a plot to visualize the total number of errors for each endpoint.
  - **Predictions Per Day**: Develop a plot showing the total number of predictions per day for each feature.
  - **Custom Plots**: Depending on specific requirements, additional plots can be designed to provide more granular insights.

Repeat these steps for each of the features in the dataset.

Report Link: https://lookerstudio.google.com/reporting/df4c3804-aec6-411e-be69-30572d6afe68 <br>
Video Link: https://youtu.be/rE-t-W4_4to





