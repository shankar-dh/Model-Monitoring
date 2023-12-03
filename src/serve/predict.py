from flask import Flask, jsonify, request
from google.cloud import storage, logging, bigquery
from google.cloud.bigquery import SchemaField
from google.api_core.exceptions import NotFound
import joblib
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.logging.type import log_severity_pb2 as severity


load_dotenv()

app = Flask(__name__)

# Set up Google Cloud logging
service_account_file = 'modelmonitoring-406915-bda8d815d315.json'
credentials = service_account.Credentials.from_service_account_file(service_account_file)
client = logging.Client(credentials=credentials)
logger = client.logger('training_pipeline')


# Initialize BigQuery client
bq_client = bigquery.Client(credentials=credentials)
table_id = os.environ['BIGQUERY_TABLE_ID']


def get_table_schema():
    """Build the table schema for the output table
    
    Returns:
        List: List of `SchemaField` objects"""
    return [
        SchemaField("CRIM", "FLOAT", mode="NULLABLE"),
        SchemaField("ZN", "FLOAT", mode="NULLABLE"),
        SchemaField("INDUS", "FLOAT", mode="NULLABLE"),
        SchemaField("CHAS", "FLOAT", mode="NULLABLE"),
        SchemaField("NOX", "FLOAT", mode="NULLABLE"),
        SchemaField("RM", "FLOAT", mode="NULLABLE"),
        SchemaField("AGE", "FLOAT", mode="NULLABLE"),
        SchemaField("DIS", "FLOAT", mode="NULLABLE"),
        SchemaField("RAD", "FLOAT", mode="NULLABLE"),
        SchemaField("TAX", "FLOAT", mode="NULLABLE"),
        SchemaField("PTRATIO", "FLOAT", mode="NULLABLE"),
        SchemaField("B", "FLOAT", mode="NULLABLE"),
        SchemaField("LSTAT", "FLOAT", mode="NULLABLE"),
        SchemaField("prediction", "FLOAT", mode="NULLABLE"),
        SchemaField("warning_flag", "INTEGER", mode="NULLABLE"),
        SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
        SchemaField("latency", "FLOAT", mode="NULLABLE"),
    ]


def create_table_if_not_exists(client, table_id, schema):
    """Create a BigQuery table if it doesn't exist
    
    Args:
        client (bigquery.client.Client): A BigQuery Client
        table_id (str): The ID of the table to create
        schema (List): List of `SchemaField` objects
        
    Returns:
        None"""
    try:
        client.get_table(table_id)  # Make an API request.
        print("Table {} already exists.".format(table_id))
    except NotFound:
        print("Table {} is not found. Creating table...".format(table_id))
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)  # Make an API request.
        print("Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id))

def initialize_variables():
    """Initialize project and bucket variables

    Returns:
        Tuple: Tuple containing project_id and bucket_name"""
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET_NAME")
    return project_id, bucket_name

def initialize_client_and_bucket(bucket_name):
    """Initialize GCS client and bucket

    Args:
        bucket_name (str): Name of the GCS bucket
    
    Returns:
        Tuple: Tuple containing storage client and bucket"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    return storage_client, bucket

def load_stats(bucket, SCALER_BLOB_NAME='scaler/normalization_stats.json'):
    scaler_blob = bucket.blob(SCALER_BLOB_NAME)
    stats_str = scaler_blob.download_as_text()
    stats = json.loads(stats_str)
    return stats

def load_min_max_values(bucket, MIN_MAX_BLOB_NAME='model/min_max_values.json'):
    min_max_blob = bucket.blob(MIN_MAX_BLOB_NAME)
    min_max_str = min_max_blob.download_as_text()
    min_max_values = json.loads(min_max_str)
    return min_max_values

def load_model(bucket, bucket_name):
    latest_model_blob_name = fetch_latest_model(bucket_name)
    local_model_file_name = os.path.basename(latest_model_blob_name)
    model_blob = bucket.blob(latest_model_blob_name)
    model_blob.download_to_filename(local_model_file_name)
    model = joblib.load(local_model_file_name)
    return model

def fetch_latest_model(bucket_name, prefix="model/model_"):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    blob_names = [blob.name for blob in blobs]
    if not blob_names:
        raise ValueError("No model files found in the GCS bucket.")
    latest_blob_name = sorted(blob_names, key=lambda x: x.split('_')[-1], reverse=True)[0]
    return latest_blob_name

def normalize_data(instance, stats):
    normalized_instance = {}
    for feature, value in instance.items():
        mean = stats["mean"].get(feature, 0)
        std = stats["std"].get(feature, 1)
        normalized_instance[feature] = (value - mean) / std
    return normalized_instance

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """Health check endpoint that returns the status of the server.
    Returns:
        Response: A Flask response with status 200 and "healthy" as the body.
    """
    return {"status": "healthy"}


@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    request_json = request.get_json()
    request_instances = request_json['instances']

    logger.log_text("Received prediction request.", severity='INFO')

    prediction_start_time = time.time()
    current_timestamp = datetime.now().isoformat()

    formatted_instances = []
    for instance in request_instances:
        out_of_range_fields = []  # List to store the names of out-of-range fields
        for feature, value in instance.items():
            feature_min_max = min_max_values.get(feature, {})
            min_val = feature_min_max.get('min', float('-inf'))
            max_val = feature_min_max.get('max', float('inf'))

            if value < min_val or value > max_val:
                out_of_range_fields.append(feature)
                logger.log_text(f"Value for {feature} in instance {instance} is out of range [{min_val}, {max_val}].", severity='WARNING')

        if out_of_range_fields:
            logger.log_text(f"Instance {instance} has out-of-range values for fields: {', '.join(out_of_range_fields)}.", severity='WARNING')

        normalized_instance = normalize_data(instance, stats)

        # Formatting instance for prediction
        formatted_instance = [
            normalized_instance.get('CRIM', 0),
            normalized_instance.get('ZN', 0),
            normalized_instance.get('INDUS', 0),
            normalized_instance.get('CHAS', 0),
            normalized_instance.get('NOX', 0),
            normalized_instance.get('RM', 0),
            normalized_instance.get('AGE', 0),
            normalized_instance.get('DIS', 0),
            normalized_instance.get('RAD', 0),
            normalized_instance.get('TAX', 0),
            normalized_instance.get('PTRATIO', 0),
            normalized_instance.get('B', 0),
            normalized_instance.get('LSTAT', 0)
        ]
        formatted_instances.append(formatted_instance)

    prediction = model.predict(formatted_instances)
    prediction_end_time = time.time()
    prediction_latency = prediction_end_time - prediction_start_time

    prediction = prediction.tolist()

    logger.log_text(f"Prediction results: {prediction}", severity='INFO')

    rows_to_insert = [
        {
            "CRIM": instance['CRIM'],
            "ZN": instance['ZN'],
            "INDUS": instance['INDUS'],
            "CHAS": instance['CHAS'],
            "NOX": instance['NOX'],
            "RM": instance['RM'],
            "AGE": instance['AGE'],
            "DIS": instance['DIS'],
            "RAD": instance['RAD'],
            "TAX": instance['TAX'],
            "PTRATIO": instance['PTRATIO'],
            "B": instance['B'],
            "LSTAT": instance['LSTAT'],
            "prediction": pred,
            "warning_flag": 1 if out_of_range_fields else 0,
            "timestamp": current_timestamp,
            "latency": prediction_latency
        }
        for instance, pred in zip(request_instances, prediction)
    ]

    errors = bq_client.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        logger.log_text("New predictions inserted into BigQuery.", severity='INFO')
    else:
        logger.log_text(f"Encountered errors inserting predictions into BigQuery: {errors}", severity='ERROR')

    output = {'predictions': [{'result': pred} for pred in prediction]}
    return jsonify(output)


project_id, bucket_name = initialize_variables()
storage_client, bucket = initialize_client_and_bucket(bucket_name)
stats = load_stats(bucket)
min_max_values = load_min_max_values(bucket)
model = load_model(bucket, bucket_name)
schema = get_table_schema()
create_table_if_not_exists(bq_client, table_id, schema)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
