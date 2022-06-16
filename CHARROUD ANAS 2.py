#!/usr/bin/env python
# coding: utf-8

# In[502]:


import numpy as np
import cv2
import os
import pandas as pd
import itertools    
import mahotas as mh
import pickle
import requests,json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import skimage.segmentation as seg
import skimage.color as color
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import skimage
from skimage.feature import greycomatrix, greycoprops
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import csv
csv.register_dialect('myDialect',quoting=csv.QUOTE_ALL,skipinitialspace=True)
from skimage.filters import prewitt_h,prewitt_v
from sklearn import preprocessing


# In[503]:


def histogram(image, mask):
    #extraire l'histogramme 3D de la region masqué d'image 
    hist1 = cv2.calcHist([image], [0, 1, 2], mask, [8,8,8],[0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist1, hist1).flatten()
    # retourner l'histogramme
    return hist
def describe(image):
    # convertire l image de RVB ====> HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    vectDescripteur = []

    # calculer le centre d'image
    (h, w) = image.shape[:2]
    
    (cX, cY) = (int(w * 0.5), int(h * 0.5))

    #diviser l'image en 4 rectangle 
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
        (0, cX, cY, h)]

    # boucler sur le segment
    for (startX, endX, startY, endY) in segments:
        # construire un masque pour chaque coin de l'image
        cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)

        #extraire l'histogramme d'image et modifier le vecteur descripteur
        hist = histogram(image, cornerMask)
        vectDescripteur.extend(hist)


    # retourner le vecteur descripteur 
    return vectDescripteur 
def textureFeatures(img):
	img = color.rgb2gray(img)
	img = skimage.img_as_ubyte(img)
	glcm = greycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
	feature = greycoprops(glcm, 'dissimilarity')[0]
	feature = np.concatenate([feature,greycoprops(glcm, 'correlation')[0]])
	feature = np.concatenate([feature,greycoprops(glcm, 'contrast')[0]])
	feature = np.concatenate([feature,greycoprops(glcm, 'energy')[0]])
	feature = feature/np.sum(feature)
	#print(feature)
	return feature


# In[504]:


#ExtraTreesClassifier
def features_normal2(features,y):
    etc = ExtraTreesClassifier(n_estimators=10)
    kk=etc.fit(features,y)
    ettc = SelectFromModel(etc, prefit=True)
    features= ettc.transform(features)
    return features
#pca
def features_normal(features):
    scaler=MinMaxScaler(feature_range=(0,1))
    pca = PCA(n_components=13)
    feat=features.astype('float')
    feat=scaler.fit_transform(feat)
    features=pca.fit_transform(feat)
    return features
#SelectKBest 
def features_normal1(features,y):
    #SelectPercentile 
    #GenericUnivariateSelect     d'autre test de reduction de nombre des features
    bestfeatures = SelectKBest(score_func=chi2, k=100)
    fit = bestfeatures.fit(abs(features),y)
    features=fit.transform(features )
    return features 


# In[505]:


names=[name for name in os.listdir("ISIC-2017_Training_Data")]
names = np.array(names)
names1=[name for name in os.listdir("ISIC-2017_Training_Part1_GroundTruth")]
names1 = np.array(names)

    


# In[506]:


names1[0]


# In[507]:


a=cv2.imread(os.path.join('ISIC-2017_Training_Data\\'+ names1[10]))
type(a)


# In[510]:


def load_features_from_folder_for_train():

    i=0
    for j in range(2000):
            
        img = cv2.imread(os.path.join('ISIC-2017_Training_Data\\'+ names[j]))
        mask1= cv2.imread(os.path.join('ISIC-2017_Training_Part1_GroundTruth\\'+ names[j]),0)
        img1= cv2.bitwise_and(img,img,mask=mask1)
        #mean,std,6
        (means, stds) = cv2.meanStdDev(img)
        mean_std=np.array(list(zip(means, stds))).flatten() 
        #shape
        gray = cv2.cvtColor(np.uint8(img1), cv2.COLOR_BGR2GRAY)
        shape=cv2.HuMoments(cv2.moments(gray)).flatten()
        shape = -np.sign(shape) * np.log10(np.abs(shape))
        #histogram
        hist =describe(img)
        #texture
        texture =textureFeatures(img)
        #feature=np.hstack((texture,shape,hist))
        #feature=np.hstack((texture))
        
       
        feature=np.hstack((hist))
        if i==0:
            features = np.zeros(feature.shape[0])
            i=i+1
        features = np.vstack((features,feature))
        
    return np.delete(features, (0), axis=0)


# In[509]:


feat_x=load_features_from_folder_for_train()


# In[472]:


feat_x.shape


# In[511]:


df = pd.read_csv('ISIC-2017_Training_Part3_GroundTruth.csv',header=None)


# In[512]:


aa=df[1][:]


# In[513]:


aa=np.array(aa)
aa=np.delete(aa, (0), axis=0)


# In[514]:


s=list(aa)


# In[515]:


names2=[name for name in os.listdir("ISIC-2017_Validation_Data")]
names2 = np.array(names2)
names3=[name for name in os.listdir("ISIC-2017_Validation_Part1_GroundTruth")]
names3 = np.array(names3)


# In[516]:


a=cv2.imread(os.path.join('ISIC-2017_Validation_Part1_GroundTruth\\'+ names3[0] ), 0)
type(a)


# In[ ]:





# In[517]:


def load_features_from_folder_for_validation():

    i=0
    for j in range(150):     
        img = cv2.imread(os.path.join('ISIC-2017_Validation_Data\\'+ names2[j]))
        mask2= cv2.imread(os.path.join('ISIC-2017_Validation_Part1_GroundTruth\\'+ names3[j] ),0)
        img1= cv2.bitwise_and(img,img,mask=mask2)
        #img = segmentation(img)
        #mean,std,6
        (means, stds) = cv2.meanStdDev(img)
        mean_std=np.array(list(zip(means, stds))).flatten() 
        #shape
        gray = cv2.cvtColor(np.uint8(img1), cv2.COLOR_BGR2GRAY)
        shape=cv2.HuMoments(cv2.moments(gray)).flatten()
        shape = -np.sign(shape) * np.log10(np.abs(shape))
        #histogram
        hist =describe(img)
        #texture
        texture =textureFeatures(img)
        #feature=np.hstack((texture,shape,hist))
        #feature=np.hstack((texture))
        
       
        feature=np.hstack((hist))
        if i==0:
            features = np.zeros(feature.shape[0])
            i=i+1
        features = np.vstack((features,feature))
        
    return np.delete(features, (0), axis=0)


# In[518]:


features1=load_features_from_folder_for_validation()


# In[519]:


df = pd.read_csv('ISIC-2017_Validation_Part3_GroundTruth.csv',header=None)


# In[520]:


bb=df[1][:]
bb=np.array(bb)
bb=np.delete(bb, (0), axis=0)


# In[521]:


s1=list(bb)


# In[522]:


# normalize the data attributes
feat_x= preprocessing.normalize(feat_x)
feat_x1=preprocessing.normalize(features1)
# standardize the data attributes
#feat_x= preprocessing.scale(feat_x)
#feat_x1= preprocessing.scale(features1)


# In[523]:


rfc =  RandomForestClassifier(n_estimators=10)
rfc.fit(feat_x,s)


# In[524]:


rfc_y=rfc.predict(feat_x1)
print("accuracy of random forrest is",accuracy_score(rfc_y,s1))
print(classification_report(rfc_y,s1))


# In[525]:


svm = SVC(C=100, gamma=0.0001, kernel='rbf',max_iter=300)
svm.fit(feat_x, s)


# In[526]:


svm_y = svm.predict(feat_x1)
print("accuracy of svm is",accuracy_score(svm_y,s1))
print(classification_report(svm_y,s1))


# In[527]:


nby=MultinomialNB()
nby.fit(abs(feat_x),s)


# In[528]:


nby_y = nby.predict(feat_x1)
print("accuracy of naive_bayes is",accuracy_score(nby_y,s1))
print(classification_report(nby_y,s1))


# In[529]:


dect = DecisionTreeClassifier()
dect.fit(feat_x, s)


# In[530]:


dect_y = dect.predict(feat_x1)
print("accuracy of decision trees is",accuracy_score(dect_y,s1))
print(classification_report(dect_y,s1))


# In[531]:


knn = KNeighborsClassifier(n_neighbors=7,weights='distance')
knn.fit(feat_x,s)

knn_y=knn.predict(feat_x1)
print("accuracy of knn is",accuracy_score(knn_y,s1))
print(classification_report(knn_y,s1))


# In[532]:


def load_features_from_folder_for_test(folder):

    i=0
    for filename in os.listdir(os.path.join(folder)):
            
        img = cv2.imread(os.path.join(os.path.join(folder),filename))
        #img = segmentation(img)
        #mean,std,6
        (means, stds) = cv2.meanStdDev(img)
        mean_std=np.array(list(zip(means, stds))).flatten() 
        #shape
        gray = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
        shape=cv2.HuMoments(cv2.moments(gray)).flatten()
        shape = -np.sign(shape) * np.log10(np.abs(shape))
        #histogram
        hist =describe(img)
        #texture
        texture =textureFeatures(img)
        #feature=np.hstack((texture,shape,hist))
        #feature=np.hstack((texture))
        
       
        feature=np.hstack((hist))
        if i==0:
            features = np.zeros(feature.shape[0])
            i=i+1
        features = np.vstack((features,feature))
        
    return np.delete(features, (0), axis=0)


# In[533]:


feat_x2=load_features_from_folder_for_test("ISIC-2017_Test_v2_Data")


# In[534]:


knn_y=knn.predict(feat_x2)
dect_y = dect.predict(feat_x2)
nby_y = nby.predict(feat_x2)
svm_y = svm.predict(feat_x2)
rfc_y=rfc.predict(feat_x2)


# In[ ]:





# In[536]:


names=[name for name in os.listdir("ISIC-2017_Test_v2_Data")]

import csv
def cret(y_pred , imgs):
    csv_file='CHARROUD ANAS12.csv'
    with open(csv_file, 'w',newline='') as csvfile:
        field=['image_id','melanoma']
        writer=csv.DictWriter(csvfile, fieldnames=field)
        writer.writeheader()
        for i in range(len(y_pred)):
            filename = imgs[i]
            classe = y_pred[i]
            writer.writerow({'image_id':filename, 'melanoma':classe})
        
cret(knn_y, names)


# In[ ]:





# In[ ]:




