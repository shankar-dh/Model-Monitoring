import pandas as pd
import json
import gcsfs
import os
from google.cloud import logging

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()

# Initialize GCP logging client
logging_client = logging.Client()
logger = logging_client.logger("data_preprocess_pipeline")


def split_data(data, split_ratio=0.8):
    """
    Splits the data into training and validation sets.
    :param data: DataFrame containing the data to be split
    :param split_ratio: The ratio to split the data by
    :return: DataFrames containing the training and validation sets
    """
    logger.log_text("Splitting the data.")
    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split the data
    split_index = int(len(data) * split_ratio)
    train_data = data.iloc[:split_index]
    val_data = data.iloc[split_index:]

    return train_data, val_data

def preprocess_data(train_data, normalization_stats_gcs_path):
    """
    Preprocesses the training data by normalizing the features and saving the normalization statistics to GCS.
    :param train_data: DataFrame containing the training data
    :param normalization_stats_gcs_path: GCS path to save the normalization statistics to
    """
    logger.log_text("Preprocessing the data.")
    train_data_numeric = train_data.drop(columns=['MEDV'])

    # Calculate mean and standard deviation for each feature in the training data
    mean_train = train_data_numeric.mean()
    std_train = train_data_numeric.std()

    # Store normalization statistics in a dictionary
    normalization_stats = {
        'mean': mean_train.to_dict(),
        'std': std_train.to_dict()
    }
    # Save the normalization statistics to a JSON file on GCS
    with fs.open(normalization_stats_gcs_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)


def main():
    logger.log_text("Main function started.")
    # Define GCS paths for the data
    train_data_gcs_path = "gs://mlops_fall2023/data/train/train_data.csv"
    test_data_gcs_path = "gs://mlops_fall2023/data/test/test_data.csv"
    normalization_stats_gcs_path = "gs://mlops_fall2023/scaler/normalization_stats.json"
    
    train_data_path = os.path.join("..", "data", "HousingData.csv")
    housing_data = pd.read_csv(train_data_path)

    # Split the data into training and validation sets
    train_data, val_data = split_data(housing_data)

    # Save the training and validation sets to CSV
    with fs.open(train_data_gcs_path, 'w') as f:
            train_data.to_csv(f, index=False)

    with fs.open(test_data_gcs_path, 'w') as f:
            val_data.to_csv(f, index=False)

    # Preprocess the training data
    preprocess_data(train_data, normalization_stats_gcs_path)



if __name__ == '__main__':
    main()
