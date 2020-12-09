# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:22:05 2020

@author: erikw
"""

import time
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from sklearn.metrics import confusion_matrix
warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline
from contextlib import redirect_stdout
from zipfile import ZipFile
import glob
import re
from tensorflow.keras.callbacks import ModelCheckpoint





class ModelSearch:
    def __init__(self):
        self.labels = None
        self.labels_summary = None
        return

    def getLabels(self):
        # First, let's store the number of classes of this particular dataset
        list_labels = [re.sub("data_template/*",'',l) for l in glob.glob("data_template/*")]
        for i in range(len(list_labels)):
            list_labels[i] = re.sub("[^a-z_A-Z]","",list_labels[i])
        print ("The labels are : " + str(list_labels))
        self.labels = list_labels
        # Check how many images per label (it will be useful later for class imbalance handling) and store that in a dictionary (also helpful for prepare train/valid/test)
        self.labels_summary = {}
        for lab in list_labels :
            to_add = {lab: len(os.listdir("data_template/"+lab+"/"))}
            self.labels_summary.update(to_add)
        print(self.labels_summary)
    
    
    def createDataDirs(self):
        for fold in ["training","validation","testing"]:
            for lab in self.labels:
                os.makedirs(fold+'/'+lab)
        # # We will use shutils.move to move the data selected randomly to the corresponding folder
        # # Take 70 % of the data for training from each class        
        for lab in self.labels:
            for i in random.sample(glob.glob('data_template/'+lab+'/*'), int(self.labels_summary[lab]*0.7)):
                shutil.move(i, 'training/'+lab)          
        # # Take 10 % of the data for validation from each class 
        for lab in self.labels:
            for i in random.sample(glob.glob('data_template/'+lab+'/*'), int(self.labels_summary[lab]*0.1)):
                shutil.move(i, 'validation/'+lab)
        # # Take the remaining 20% for testing
        for lab in self.labels:
            for i in glob.glob('data_template/'+lab+'/*'):
                shutil.move(i, 'testing/'+lab)
                
    
    def plot_confusion_matrix(self,cm, classes,name,
                          cmap=plt.cm.Blues,save=True):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if save :
            plt.savefig("./"+name)
        
        
    def plot_history(self,history,name="",save=True):
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
        ## As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
        ## Loss
        plt.figure(2)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if save:
            plt.savefig("./"+name+"_loss.png")
        plt.legend()
        ## Accuracy
        plt.figure(3)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if save :
            plt.savefig("./"+name+"_accuracy.png")
    
    
    def findModel(self):
        #max val acccuracy, epoch at max val_acc, model type of max val_acc
        mx = 0
        epochAtMx = 0
        modelType = ""
        #Set initial hyperparameters for training runs
        batch_size=16
        n_epochs=10
        lr = 0.001
        models = ['vgg16','mobilenet','inception','inception_resnet','xception']
        #create directory for saved model files with highest val_accuracy
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        #model params to set for each training run
        preprocessingFunc = None
        model = None
        removeLayer = 0
        freezeLayers = 0
        #batches to set for each training run
        train_batches = None
        valid_batches = None
        test_batches = None
        
        #Perform training for each model type, saving the best models and validation accuracies.
        #Record best overall val_accuracy,epoch,model
                        #len(models)
        for i in range(len(models)):
            model_name = models[i]
            #callback for saving checkpoint at best validation accuracy
            model_filename = '%s_model.{epoch:03d}.h5' % model_name
            filepath = os.path.join(save_dir, model_filename)
            # prepare callback for model saving
            checkpoint = ModelCheckpoint(filepath=filepath,
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True)
              
            if model_name == 'vgg16':
                preprocessingFunc = tf.keras.applications.vgg16.preprocess_input
                model = tf.keras.applications.vgg16.VGG16()
                removeLay = -2
                freezeLay = -3
            elif model_name == 'mobilenet':
                preprocessingFunc = tf.keras.applications.mobilenet.preprocess_input
                model = tf.keras.applications.mobilenet.MobileNet()
                removeLay = -6
                freezeLay = -5
            elif model_name == 'inception':
                preprocessingFunc = tf.keras.applications.inception_v3.preprocess_input
                model = tf.keras.applications.inception_v3.InceptionV3()
                removeLay = -1
                freezeLay = -5
            elif model_name == 'inception_resnet':
                preprocessingFunc = tf.keras.applications.inception_resnet_v2.preprocess_input
                model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2()
                removeLay = -1
                freezeLay = -5
            elif model_name == 'xception':
                preprocessingFunc = tf.keras.applications.xception.preprocess_input
                model = tf.keras.applications.xception.Xception()
                removeLay = -1
                freezeLay = -5
            
            train_batches = ImageDataGenerator(preprocessing_function=preprocessingFunc) .flow_from_directory(directory="training/", target_size=(224,224), classes=self.labels, batch_size=batch_size)
            valid_batches = ImageDataGenerator(preprocessing_function=preprocessingFunc) .flow_from_directory(directory="validation/", target_size=(224,224), classes=self.labels, batch_size=batch_size)
            test_batches = ImageDataGenerator(preprocessing_function=preprocessingFunc) .flow_from_directory(directory="testing", target_size=(224,224), classes=self.labels, batch_size=batch_size, shuffle=False)
            
            x = model.layers[removeLay].output  # we remove the output. layer that we  replace with the following line
            output = Dense(units=len(self.labels), activation='softmax')(x)
            model_new = Model(inputs=model.input, outputs=output)
            for layer in model_new.layers[:freezeLay]:  # retrain fc1 , fc2 and prediction layer
                layer.trainable = False
            model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            #model_new.summary()
            history_model = model_new.fit(x=train_batches,
                        validation_data=valid_batches,
                        validation_steps=len(valid_batches),
                        epochs=n_epochs,
                        verbose=1,callbacks=[checkpoint]
            )
            #print the max validation accuracy and the epoch at which it occured
            for k in range(len(history_model.history['val_accuracy'])):
                if history_model.history['val_accuracy'][k] > mx:
                    mx = history_model.history['val_accuracy'][k]
                    epochAtMx = k + 1
                    modelType = model_name
            print('max val acc: {}'.format(mx))
            print('at epoch: {}'.format(epochAtMx))
            print('best model: {}'.format(modelType))
            
        #Select and load the model with highest recorded val_accuracy
        bestModelFile = modelType + ('_model.%03d.h5' % epochAtMx)
        bestModel = keras.models.load_model('saved_models/'+bestModelFile)
        print('best model file: {}'.format(bestModelFile))
        
        #Set model attributes according to best model type
        if modelType == 'vgg16':
            preprocessingFunc = tf.keras.applications.vgg16.preprocess_input
            removeLay = -2
            freezeLay = -3
        elif modelType == 'mobilenet':
            preprocessingFunc = tf.keras.applications.mobilenet.preprocess_input
            removeLay = -6
            freezeLay = -5
        elif modelType == 'inception':
            preprocessingFunc = tf.keras.applications.inception_v3.preprocess_input
            removeLay = -1
            freezeLay = -5
        elif modelType == 'inception_resnet':
            preprocessingFunc = tf.keras.applications.inception_resnet_v2.preprocess_input
            removeLay = -1
            freezeLay = -5
        elif modelType == 'xception':
            preprocessingFunc = tf.keras.applications.xception.preprocess_input
            removeLay = -1
            freezeLay = -5
        
        #Initialize new callback for best model
        #callback for saving checkpoint at best validation accuracy
        model_name = 'bestModel'
        model_filename = '%s_model.{epoch:03d}.h5' % model_name
        filepath = os.path.join(save_dir, model_filename)
        # prepare callback for model saving
        checkpoint = ModelCheckpoint(filepath=filepath,
                                  monitor='val_accuracy',
                                  verbose=1,
                                  save_best_only=True)
        
        #Update hyperparameters for tuning the best model
        #Create new batches for best model.
        #Then start training.
        #set new number of epochs , batch size, lern rate for best model tuning
        n_epochs=15
        batch_size = 32
        lr = 0.00001
        #create batches for new training round with updated batch size
        train_batches = ImageDataGenerator(rotation_range=5,horizontal_flip=True,preprocessing_function=preprocessingFunc) .flow_from_directory(directory="training/", target_size=(224,224), classes=self.labels, batch_size=batch_size)
        valid_batches = ImageDataGenerator(rotation_range=5,horizontal_flip=True,preprocessing_function=preprocessingFunc) .flow_from_directory(directory="validation/", target_size=(224,224), classes=self.labels, batch_size=batch_size)
        test_batches = ImageDataGenerator(preprocessing_function=preprocessingFunc) .flow_from_directory(directory="testing", target_size=(224,224), classes=self.labels, batch_size=batch_size, shuffle=False)
        opt = keras.optimizers.Adam(learning_rate=lr)
        bestModel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_bestModel = bestModel.fit(x=train_batches,
                    steps_per_epoch=len(train_batches),
                    validation_data=valid_batches,
                    validation_steps=len(valid_batches),
                    epochs=n_epochs,
                    verbose=1,callbacks=[checkpoint]
        )
        
        #Record best model's top validation accuracy and epoch.
        #Model h5 file saved by checkpoint callback.
        #print the max validation accuracy and the epoch at which it occured
        for k in range(len(history_bestModel.history['val_accuracy'])):
            if history_bestModel.history['val_accuracy'][k] > mx:
                mx = history_bestModel.history['val_accuracy'][k]
                epochAtMx = k + 1
                modelType = model_name
        print('max val acc: {}'.format(mx))
        print('at epoch: {}'.format(epochAtMx))
        print('best model: {}'.format(modelType))
        
        #save model summary
        summary_file = 'bestModel_summary.txt'
        with open(summary_file, 'w') as f:
            with redirect_stdout(f):
                bestModel.summary()
        
        #Show confusion matrix and training history of best model training run with tuned hyperparameters.
        predictions_bestModel = bestModel.predict(x=test_batches, steps=len(test_batches), verbose=0) 
        cm = confusion_matrix(y_true=test_batches.classes, y_pred=predictions_bestModel.argmax(axis=1))
        self.plot_confusion_matrix(cm=cm, classes=list(test_batches.class_indices.keys()),name="confusion_matrix_bestModel.png")
        plt.show()
        self.plot_history(history_bestModel,name="history_bestModel")
        plt.show()
        return modelType, epochAtMx

        

def main():
    ms = ModelSearch()
    ms.getLabels()
    ms.createDataDirs()
    ms.findModel()
    
    
if __name__ == "__main__":
    main()
        
        