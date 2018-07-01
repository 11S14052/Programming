# -*- coding: utf-8 -*-
"""
Created on Mon May  7 20:09:53 2018

@author: Wahyu Nainggolan
"""

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Normalizer
import cv2
import os
import sys 
import numpy as np
#from sklearn.metrics.classification import accuracy_score
sys.path.append(os.path.abspath("C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/Face_recognation with LBPH and DBN/"))
from dbn.tensorflow import  SupervisedDBNClassification


#preprocessing data
def face_recognation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:/Users/Wahyu Nainggolan/Documents/kuliahku\Semester 8/Face_recognation with LBPH and DBN/opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.01,minNeighbors=3)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

#keterangan :
                #1. tipe data harus jpg
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
             if 'dosen.' in image_name:
               labels.append(0)
             else :
                if  "mahasiswa." in image_name:
                    labels.append(1)
                else : 
                    if "pegawai." in image_name:
                        labels.append(2)
                    else :
                        labels.append(3)   
             image_path = subject_dir_path + "/" + image_name
             image2 = cv2.imread(image_path)
             cv2.imshow("Training on image...", cv2.resize(image2, (400, 500)))
             cv2.waitKey(100)
             face, rect = face_recognation(image2)
             face=cv2.resize(face, (28,28))
             face = face.astype(np.float32)/(np.iinfo(face.dtype)).max
             faces.append(np.ravel(face))
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/Face_recognation with LBPH and DBN/training-data")
print("Data prepared")

faces = np.array(faces, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)
print("Total faces/train: ", faces.shape)
print("Total labels/test: ", labels.shape)

##########Normalisasi #############################
norm=Normalizer()
X=norm.fit_transform(faces)
X = (faces / 32).astype(np.float32)
Y=labels
X=faces

############################ K-fold Cross Validation ####################################
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix  
acc_train=[]
acc_test=[]
train_loss=[]
validation_loss=[]
pred_train=[]
pred_test=[]
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
_features = np.asarray(X)
_labels = np.asarray(Y)
no =0
mdl=0
for train, test in kfold.split(X, Y):
    # Training
    x_train = np.asarray(X)[train]
    y_train = np.asarray(Y)[train]
    x_test = np.asarray(X)[test]
    y_test = np.asarray(Y)[test]

    mdl+=1
    print("Ukuran data model ",mdl)
    print("Ukuran x_train : ",x_train.shape)
    print("Ukuran y_train",y_train.shape)
    print("Ukuran x_test : ",x_test.shape)
    print("Ukuran y_test",y_test.shape)

    classifier = SupervisedDBNClassification(hidden_layers_structure=[len(np.asarray(X)[train]),len(np.asarray(Y)[train])],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
    tr_loss=classifier.fit(np.asarray(X)[train], np.asarray(Y)[train])
    val_loss = classifier.fit(np.asarray(X)[test], np.asarray(Y)[test])
    train_loss.append(tr_loss)
    validation_loss.append(val_loss)
    predict_train = classifier.predict(np.asarray(X)[train])
    accuracy_train = accuracy_score(np.asarray(Y)[train], predict_train)
    acc_train.append(accuracy_train)    
    pred_train.append(predict_train)
    predict_test = classifier.predict(np.asarray(X)[test])
    pred_test.append(predict_test)
    accuracy_test = accuracy_score(np.asarray(Y)[test], predict_test)
    acc_test.append(accuracy_test)
    #predict=classifier.predict(np.asarray(X)[test])
    no+=1
    print('Pada model {0} Accuracy Train adalah : {1} %'.format (no,accuracy_train))    
    print('Pada model {0} Accuracy Test adalah: {1} %'.format (no,accuracy_test))    
    print('Pada model {0} berikut klasifikasi dari hasilnya: {1} '.format(no,classification_report(Y_test, Y_pred)))      


print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))  

print(classification_report(Y_test, Y_pred))  

plt.plot(acc_train,color='black')
plt.plot(acc_test,color='red')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Accuracy Train', 'Accuracy Validation'], loc='upper left')
plt.show()




#######################################
########## Save model train ######################
#classifier.save('C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/projek_real/model_saver/model.pkl')
#classifier.save('C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/projek_real/model_saver/model.pkl')
#from sklearn.metrics.classification import accuracy_score
#print('Done.\nAccuracy: %f' % accuracy_score(Y_validation, b))

            


subjects = ["Dosen", "Mahasiswa","Pegawai","Bukan Civitas Del"]
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    

b=[]    
c=[]
def predict(test_img):
    face, rect = face_recognation(test_img)
    face=cv2.resize(face, (28,28))    
    face_predict=np.ravel(face)
    b.append(face_predict)
    label= classifier.predict(b)
    label = np.array(label, dtype=np.int32)

    label_text = subjects[label[0]]
    #draw_rectangle(test_img, rect)  
    draw_text(test_img, label_text, rect[0], rect[1]-5)
    
    return test_img    


print("Predicting images...")


#load test images
test_img1 = cv2.imread("C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/projek_real/data_test/mahasiswa.27.jpg")
test_img2 = cv2.imread("C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/projek_real/data_test/mahasiswa.176.jpg")
test_img3 = cv2.imread("C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/projek_real/data_test/bukancivitas.1.jpg")
test_img4 = cv2.imread("C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/projek_real/data_test/mahasiswa.10.jpg")
test_img5 = cv2.imread("C:/Users/Wahyu Nainggolan/Documents/kuliahku/Semester 8/projek_real/data_test/mahasiswa.11.jpg")

#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
predicted_img5 = predict(test_img5)

print("Prediction complete")


#display both images
cv2.imshow("Hasil Prediksinya adalah...",  cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(3000)
cv2.destroyAllWindows()

cv2.imshow("Hasil Prediksinya adalah...",  cv2.resize(predicted_img1, (400, 500)))
cv2.waitKey(3000)
cv2.destroyAllWindows()

cv2.imshow("Hasil Prediksinya adalah...",  cv2.resize(predicted_img3, (400, 500)))
cv2.waitKey(3000)
cv2.destroyAllWindows()

cv2.imshow("Hasil Prediksinya adalah...",  cv2.resize(predicted_img4, (400, 500)))
cv2.waitKey(3000)
cv2.destroyAllWindows()

cv2.imshow("Hasil Prediksinya adalah...",  cv2.resize(predicted_img5, (400, 500)))
cv2.waitKey(3000)
cv2.destroyAllWindows()




#########################################
###########Visualisasi data #############
##########################################
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(faces, bins=100)
plt.title("Distribution of The X Values")
plt.xlabel("Data Range")
plt.ylabel("Frequency")
plt.show()
###############################
plt.hist(X, bins=100)
plt.title("Distribution of The X Values")
plt.xlabel("Data Range")
plt.ylabel("Frequency")
plt.show()
##################################
