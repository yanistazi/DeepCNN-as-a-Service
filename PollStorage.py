# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:26:27 2020


"""


from google.cloud import storage
import os
import time
import re
from zipfile import ZipFile
from pathlib import Path
import shutil

class PollStorage:
    def __init__(self):
        self.email_add = None
        return
    
    def start_polling(self):
        #set google credentials and init google storage client
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']="[USER KEY]"
        client = storage.Client()
        blobList = []
        datasetName = ''
        #poll dataset bucket every n seconds to check for blobs
        while True:
            blobList = client.list_blobs("[DATASET BUCKET]", prefix="dataset")
            for blob in blobList:
                datasetName = blob.name
                break
            if datasetName != '':
                break
            print('no files found in bucket')
            time.sleep(5)
    
        #get timestamp of file
        datasetTime = re.findall("[0-9]{6}", datasetName)[0]
        
        #get the corresponding email from email bucket and notify
        email_fname = 'emailToDeliver.txt'
        email_bucket = client.bucket('[EMAIL BUCKET NAME]')
        email_blob = email_bucket.blob('email' + datasetTime + '.txt')
        email_blob.download_to_filename(email_fname)
        emailFile = open(email_fname, 'r')
        emailAddress = emailFile.readline()
        self.email_add = emailAddress
        emailFile.close()
        print('file was found, will send report to {} when finished processing'.format(emailAddress))
        
        #download dataset for processing
        extension = os.path.splitext(datasetName)[1]
        data_fname = 'dataset' + extension
        data_bucket = client.bucket('[DATASET BUCKET NAME]')
        data_blob = data_bucket.blob(datasetName)
        data_blob.download_to_filename(data_fname)
        print('dataset downloaded to {}, start processing'.format(data_fname))
        #delete email and dataset items in cloud storage
        email_blob.delete()
        data_blob.delete()
        print("email and data blobs deleted.")
        
        #delete any pre-existing directory data_template/
        dirPath = Path('data_template')
        if dirPath.exists() and dirPath.is_dir():
            shutil.rmtree(dirPath)
        
        #extract files from dataset.zip to data_template/
        with ZipFile(data_fname,'r') as zip_file:
            zip_file.extractall('data_template')
            
    def get_email(self):
        return self.email_add


def main():
    poll = PollStorage()
    poll.start_polling()
    

if __name__ == "__main__":
    main()




