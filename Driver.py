# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:18:20 2020


"""

from PollStorage import PollStorage
from StoreModel import StoreModel
from SendReport import SendReport
from ModelSearch import ModelSearch


#poll the google storage bucket for user input
poll = PollStorage()
poll.start_polling()

#start searching for model / training
ms = ModelSearch()
ms.getLabels()
ms.createDataDirs()
modelType, epochAtMx = ms.findModel() #returns best model and epoch at max val acc


model_file = '%s_model.{:03d}.h5'.format(epochAtMx) % modelType
storeModel = StoreModel()
storeModel.store_model(local_file='./saved_models/{}'.format(model_file))
print(storeModel.get_model_link())
                                                                            
sendReport = SendReport(model_link=storeModel.get_model_link(),email_add=poll.get_email() \
                        ,cm_img='confusion_matrix_bestModel.png' \
                        ,loss_img='history_bestModel_loss.png' \
                        ,acc_img='history_bestModel_accuracy.png' \
                        ,model_sum='bestModel_summary.txt')
sendReport.sendEmail()
