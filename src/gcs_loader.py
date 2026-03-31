from google.cloud import storage
import pandas as pd
import os

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)

    print(f"Downloaded {source_blob_name} to {destination_file_name}")

# Example usage
if __name__ == "__main__":
    bucket_name = "product-recommender-systems"
    
    download_from_gcs(
        bucket_name,
        "events.csv",
        "data/raw/events.csv"
    )