import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import gcsfs
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import logging
from google.oauth2 import service_account
from google.logging.type import log_severity_pb2 as severity


service_account_file = '/trainer/modelmonitoring-406915-bda8d815d315.json'
credentials = service_account.Credentials.from_service_account_file(service_account_file)
client = logging.Client(credentials=credentials)
logger = client.logger('training_pipeline')

# Load environment variables
load_dotenv()

# Initialize variables
fs = gcsfs.GCSFileSystem()
storage_client = storage.Client()
bucket_name = os.getenv("BUCKET_NAME")
MODEL_DIR = os.getenv("AIP_MODEL_DIR")




def load_data(gcs_train_data_path):
    logger.log_text(f"Loading training data from {gcs_train_data_path}.")
    with fs.open(gcs_train_data_path) as f:
        df = pd.read_csv(f)
    return df

def normalize_data(data, stats):
    logger.log_text("Normalizing the data.")
    normalized_data = {}
    for column in data.columns:
        mean = stats["mean"][column]
        std = stats["std"][column]
        normalized_data[column] = [(value - mean) / std for value in data[column]]
    normalized_df = pd.DataFrame(normalized_data, index=data.index)
    return normalized_df

def calculate_min_max(df):
    min_max_values = df.agg(['min', 'max']).to_dict()
    return min_max_values

def save_min_max_to_gcs(min_max_values, gcs_path):
    logger.log_text(f"Saving min-max values to {gcs_path}.")
    min_max_json = json.dumps(min_max_values)
    with fs.open(gcs_path, 'w') as f:
        f.write(min_max_json)

def data_transform(df):
    logger.log_text("Transforming the data.")
    blob_path = 'scaler/normalization_stats.json'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_string()
    stats = json.loads(data)
    df = df.fillna(df.median())
    X_train = df.drop(columns=['MEDV'])
    y_train = df['MEDV']

    # Normalize the data
    X_train = normalize_data(X_train, stats)
    return X_train, y_train

def train_model(X_train, y_train):
    logger.log_text("Training the Random Forest Regressor model.")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_and_upload_model(model, local_model_path, gcs_model_path):
    logger.log_text(f"Saving and uploading the model to {gcs_model_path}.")
    joblib.dump(model, local_model_path)
    with fs.open(gcs_model_path, 'wb') as f:
        joblib.dump(model, f)
def main():
    # Load data
    gcs_train_data_path = "gs://mlops_fall2023/data/train/train_data.csv"
    df = load_data(gcs_train_data_path)

    # Fill missing values with median
    df = df.fillna(df.median())

    # Calculate and save min and max values before normalization
    min_max_values = calculate_min_max(df)
    min_max_gcs_path = f"{MODEL_DIR}/min_max_values.json"
    save_min_max_to_gcs(min_max_values, min_max_gcs_path)

    # Drop target column and normalize data
    X_train = df.drop(columns=['MEDV'])
    y_train = df['MEDV']
    
    # Load normalization stats and normalize data
    blob_path = 'scaler/normalization_stats.json'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_string()
    stats = json.loads(data)
    X_train = normalize_data(X_train, stats)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the model locally and upload to GCS
    edt = pytz.timezone('US/Eastern')
    current_time_edt = datetime.now(edt)
    version = current_time_edt.strftime('%Y%m%d_%H%M%S')
    local_model_path = "model.pkl"
    gcs_model_path = f"{MODEL_DIR}/model_{version}.pkl"
    print(gcs_model_path)
    save_and_upload_model(model, local_model_path, gcs_model_path)

if __name__ == "__main__":
    main()

