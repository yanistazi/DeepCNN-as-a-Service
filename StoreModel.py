# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:18:38 2020

@author: erikw
"""
#curl -X PUT -H 'Content-Type: application/octet-stream' --upload-file cars.csv $(curl https://us-central1-dl-project-292616.cloudfunctions.net/uploadDatasetCSV?email=erik0@yahoo.com)

from google.cloud import storage
import os
import io
import time
from datetime import datetime as dt


class StoreModel:
    def __init__(self):
        self.model_link = None
        return
    def store_model(self, local_file):
        #set google credentials and init google storage client
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']="googleServiceAccountKey.json"
        client = storage.Client()
        sourceFile = local_file
        destFile = 'model' + dt.now().strftime('%H%M%S') + '.h5'
        fileLink = 'https://storage.googleapis.com/stored-models/' + destFile
        #upload model file to google storage
        if self.upload_blob('stored-models', sourceFile, destFile):
            self.model_link = fileLink
        
    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # bucket_name = "bucket-name"
        # source_file_name = "local/path/to/file"
        # destination_blob_name = "storage-object-name"
    
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
    
        blob.upload_from_filename(source_file_name)
    
        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )
        return True
    
    def get_model_link(self):
        return self.model_link
    

def main():
    storeModel = StoreModel()
    storeModel.store_model(local_file='./saved_models/model.txt')
    print(storeModel.get_model_link())
    
if __name__ == "__main__":
    main()
    
    