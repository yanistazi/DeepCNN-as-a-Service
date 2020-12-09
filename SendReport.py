# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:23:34 2020

@author: erikw
"""

import smtplib, ssl

from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

#params: link to file , email address, conf matrix image, loss image, acc image, model summary text

class SendReport:
    def __init__(self, model_link, email_add, cm_img, loss_img, acc_img, model_sum):
        self.model_link = model_link
        self.email_add = email_add
        self.cm_img = cm_img
        self.loss_img = loss_img
        self.acc_img = acc_img
        self.model_sum = model_sum
        
    def sendEmail(self):
        msg = EmailMessage()
        msg['Subject'] = 'Your trained and optimized model is ready'
        msg['From'] = "dlprojectnyu@gmail.com"
        msg['To'] = self.email_add
        
        #attach the model summary text file
        with open(self.model_sum, 'rb') as content_file:
            content = content_file.read()
            msg.add_attachment(content, maintype='application', subtype='txt', filename='model_summary.txt')
        #embed images of accuracy,loss,confMatrix in the email
        with open(self.acc_img, 'rb') as content_file:
            content = content_file.read()
            msgImage = MIMEImage(content)
            msgImage.add_header('Content-ID','<acc_image>')
            msg.attach(msgImage)
        with open(self.loss_img, 'rb') as content_file:
            content = content_file.read()
            msgImage = MIMEImage(content)
            msgImage.add_header('Content-ID','<loss_image>')
            msg.attach(msgImage)
        with open(self.cm_img, 'rb') as content_file:
            content = content_file.read()
            msgImage = MIMEImage(content)
            msgImage.add_header('Content-ID','<cm_image>')
            msg.attach(msgImage)
            
        
        #set body of email
        msgTextHeader = MIMEText('You can download your Keras model h5 file here: \n ' + self.model_link + '\n\n')
        msg.attach(msgTextHeader)
        #reference the images in the IMG SRC attribute by the IDs
        msgText = MIMEText('Report of your optimized model\'s performance in terms of accuracy, loss, and confusion matrix<br><br><br> \
            <img src="cid:acc_image"><br><br><img src="cid:loss_image"><br><br><img src="cid:cm_image"><br><br>','html')
        msg.attach(msgText)
        
        port = 465 #for SSL
        
        #create a secure SSL context
        context = ssl.create_default_context()
        password = 'projectDL2020!'
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login("dlprojectnyu@gmail.com", password)
            server.send_message(msg)
            server.quit()
            

def main():
    '''
    best_model = 'scratch'
    best_model2 = 'from_scratch'
    sendReport = SendReport(model_link="https://www.google.com",email_add='ewei485@gmail.com' \
                        ,cm_img='user_output/confusion_matrix_'+best_model+'.png' \
                        ,loss_img='user_output/history_'+best_model+'_loss.png' \
                        ,acc_img='user_output/history_'+best_model+'_accuracy.png' \
                        ,model_sum='user_output/model_'+best_model2+'_summary.txt')
    sendReport.sendEmail()
    '''
    
    sendReport = SendReport(model_link='https://www.google.com',email_add='ewei485@gmail.com' \
                            , cm_img='google_play_download.jpeg',loss_img='google_play_download.jpeg' \
                                ,acc_img='google_play_download.jpeg',model_sum='user_output/model.txt')
    sendReport.sendEmail()
    
if __name__ == "__main__":
    main()
    

    

